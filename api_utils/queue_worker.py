"""
Queue Worker Module
Handles tasks in the request queue.
"""

import asyncio
import logging
import time
from asyncio import Event, Future, Lock, Queue, Task
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from playwright.async_api import Locator

from api_utils.context_types import QueueItem
from logging_utils import set_request_id, set_source
from models import ChatCompletionRequest

from .error_utils import (
    client_cancelled,
    client_disconnected,
    processing_timeout,
    server_error,
)


class QueueManager:
    def __init__(self):
        self.logger = logging.getLogger("queue_worker")
        self.was_last_request_streaming = False
        self.last_request_completion_time = 0.0

        # These will be initialized from server.py or created if missing
        self.request_queue: Optional[Queue[QueueItem]] = None
        self.processing_lock: Optional[Lock] = None
        self.model_switching_lock: Optional[Lock] = None
        self.params_cache_lock: Optional[Lock] = None

        # Context for cleanup
        self.current_submit_btn_loc: Optional[Locator] = None
        self.current_client_disco_checker: Optional[Callable[[str], bool]] = None
        self.current_completion_event: Optional[Event] = None
        self.current_req_id: Optional[str] = None

    def initialize_globals(self) -> None:
        """Initialize global variables from server state or create new ones."""
        from api_utils.server_state import state

        # Use state's logger if available, otherwise keep local one
        if hasattr(state, "logger"):
            self.logger = state.logger

        self.logger.info("--- Queue Worker Initializing ---")

        if state.request_queue is None:
            self.logger.info("Initializing request_queue...")
            state.request_queue = Queue()
        self.request_queue = state.request_queue

        if state.processing_lock is None:
            self.logger.info("Initializing processing_lock...")
            state.processing_lock = Lock()
        self.processing_lock = state.processing_lock

        if state.model_switching_lock is None:
            self.logger.info("Initializing model_switching_lock...")
            state.model_switching_lock = Lock()
        self.model_switching_lock = state.model_switching_lock

        if state.params_cache_lock is None:
            self.logger.info("Initializing params_cache_lock...")
            state.params_cache_lock = Lock()
        self.params_cache_lock = state.params_cache_lock

    async def check_queue_disconnects(self) -> None:
        """Check for disconnected clients in the queue."""
        if not self.request_queue:
            return

        queue_size = self.request_queue.qsize()
        if queue_size == 0:
            return

        checked_count = 0
        items_to_requeue: List[QueueItem] = []
        processed_ids: Set[str] = set()

        # Limit check to 10 items or queue size
        limit = min(queue_size, 10)

        while checked_count < limit:
            try:
                item: QueueItem = self.request_queue.get_nowait()
                item_req_id = str(item.get("req_id", "unknown"))

                if item_req_id in processed_ids:
                    items_to_requeue.append(item)
                    continue

                processed_ids.add(item_req_id)

                if not item.get("cancelled", False):
                    item_http_request = item.get("http_request")
                    if item_http_request:
                        try:
                            if await item_http_request.is_disconnected():
                                set_request_id(item_req_id)
                                self.logger.info(
                                    "(Worker Queue Check) Client disconnected, marking cancelled."
                                )
                                item["cancelled"] = True
                                item_future = item.get("result_future")
                                if item_future and not item_future.done():
                                    item_future.set_exception(
                                        client_disconnected(
                                            item_req_id,
                                            "Client disconnected while queued.",
                                        )
                                    )
                        except asyncio.CancelledError:
                            raise
                        except Exception as check_err:
                            set_request_id(item_req_id)
                            self.logger.error(
                                f"(Worker Queue Check) Error checking disconnect: {check_err}"
                            )

                items_to_requeue.append(item)
                checked_count += 1
            except asyncio.QueueEmpty:
                break

        for item in items_to_requeue:
            await self.request_queue.put(item)

    async def get_next_request(self) -> Optional[QueueItem]:
        """Get the next request from the queue with timeout."""
        if not self.request_queue:
            await asyncio.sleep(1)
            return None

        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return None

    async def handle_streaming_delay(
        self, req_id: str, is_streaming_request: bool
    ) -> None:
        """Handle delay between streaming requests."""
        current_time = time.time()
        if (
            self.was_last_request_streaming
            and is_streaming_request
            and (current_time - self.last_request_completion_time < 1.0)
        ):
            delay_time = max(
                0.5, 1.0 - (current_time - self.last_request_completion_time)
            )
            self.logger.info(
                f"(Worker) Sequential streaming request, adding {delay_time:.2f}s delay..."
            )
            await asyncio.sleep(delay_time)

    async def process_request(self, request_item: QueueItem) -> None:
        """Process a single request item."""
        req_id = str(request_item["req_id"])
        request_data: ChatCompletionRequest = request_item["request_data"]
        http_request: Request = request_item["http_request"]
        result_future: Future[Union[StreamingResponse, JSONResponse]] = request_item[
            "result_future"
        ]

        # 设置日志上下文 (Grid Logger)
        set_request_id(req_id)
        set_source("WORKER")

        # 1. Check cancellation
        if request_item.get("cancelled", False):
            self.logger.info("(Worker) Request cancelled, skipping.")
            if not result_future.done():
                result_future.set_exception(
                    client_cancelled(req_id, "Request cancelled by user")
                )
            if self.request_queue:
                self.request_queue.task_done()
            return

        is_streaming_request = bool(request_data.stream)
        self.logger.info(
            f"Processing request (Stream={'Yes' if is_streaming_request else 'No'})"
        )

        # 2. Initial Connection Check
        from api_utils.request_processor import (
            _check_client_connection,  # pyright: ignore[reportPrivateUsage]
        )

        if not await _check_client_connection(req_id, http_request):
            self.logger.info("(Worker) Client disconnected before processing.")
            if not result_future.done():
                result_future.set_exception(
                    HTTPException(
                        status_code=499,
                        detail=f"[{req_id}] Client disconnected before processing",
                    )
                )
            if self.request_queue:
                self.request_queue.task_done()
            return

        # 3. Streaming Delay
        await self.handle_streaming_delay(req_id, is_streaming_request)

        # 4. Connection Check before Lock
        if not await _check_client_connection(req_id, http_request):
            self.logger.info("(Worker) Client disconnected while waiting.")
            if not result_future.done():
                result_future.set_exception(
                    HTTPException(
                        status_code=499, detail=f"[{req_id}] Client disconnected"
                    )
                )
            if self.request_queue:
                self.request_queue.task_done()
            return

        self.logger.info("(Worker) Waiting for processing lock...")

        if not self.processing_lock:
            self.logger.error("Processing lock is None!")
            if not result_future.done():
                result_future.set_exception(
                    server_error(req_id, "Internal error: Processing lock missing")
                )
            if self.request_queue:
                self.request_queue.task_done()
            return

        async with self.processing_lock:
            self.logger.info("(Worker) Acquired processing lock.")

            # 5. Final Connection Check inside Lock
            if not await _check_client_connection(req_id, http_request):
                self.logger.info("(Worker) Client disconnected inside lock.")
                if not result_future.done():
                    result_future.set_exception(
                        HTTPException(
                            status_code=499, detail=f"[{req_id}] Client disconnected"
                        )
                    )
            elif result_future.done():
                self.logger.info("(Worker) Future already done. Skipping.")
            else:
                # --- Fast-Fail Tiered Error Recovery Logic ---
                # 优化后的恢复策略 (目标: 10-15秒内切换失败配置文件)
                # Tier 1: Page Refresh (快速，~2-3秒)
                # Tier 2: Auth Profile Switch (跳过浏览器重启，直接切换配置文件)

                max_attempts = 3  # Attempt 1 (Initial) -> Tier 1 (Refresh) -> Attempt 2 -> Tier 2 (Profile Switch) -> Attempt 3

                for attempt in range(1, max_attempts + 1):
                    try:
                        # 检查 result_future 是否已完成 (可能在之前的尝试中已设置结果/异常)
                        # asyncio.Future 无法重置，一旦完成就不能有意义地重试
                        if result_future.done():
                            self.logger.warning(
                                f"(Worker) [Attempt {attempt}] result_future already done, "
                                "cannot retry with same future. Breaking retry loop."
                            )
                            break

                        self.logger.info(
                            f"(Worker) [Attempt {attempt}/{max_attempts}] Executing request logic..."
                        )
                        await self._execute_request_logic(
                            req_id, request_data, http_request, result_future
                        )
                        # If successful (no exception raised), break the retry loop
                        break
                    except asyncio.CancelledError:
                        # Check if this is a user-initiated shutdown
                        from api_utils.server_state import state

                        if state.should_exit:
                            self.logger.info("Worker stopped by user.")
                        else:
                            self.logger.warning(
                                "(Worker) Request cancelled during execution."
                            )
                        raise
                    except Exception as e:
                        error_str = str(e).lower()
                        self.logger.error(
                            f"(Worker) [Attempt {attempt}/{max_attempts}] Error: {e}"
                        )

                        # 检查是否为配额错误 - 立即切换配置文件
                        is_quota_error = any(
                            keyword in error_str
                            for keyword in [
                                "quota",
                                "429",
                                "rate limit",
                                "exceeded",
                                "too many requests",
                            ]
                        )

                        if is_quota_error:
                            self.logger.warning(
                                "(Worker) 检测到配额/限流错误，立即切换配置文件..."
                            )
                            await self._switch_auth_profile(req_id)
                            continue

                        # If it's the last attempt, re-raise to be handled by outer block
                        if attempt == max_attempts:
                            self.logger.critical(
                                f"(Worker) All {max_attempts} attempts failed."
                            )
                            raise

                        # Tier 1: Page Refresh (快速恢复，~2-3秒)
                        if attempt == 1:
                            self.logger.info(
                                "(Worker) Tier 1 Recovery: Page Refresh..."
                            )
                            try:
                                await self._refresh_page(req_id)
                            except asyncio.CancelledError:
                                raise
                            except Exception as refresh_err:
                                self.logger.error(
                                    f"(Worker) Tier 1 Refresh failed: {refresh_err}"
                                )
                            continue

                        # Tier 2: Auth Profile Switch (跳过浏览器重启，直接切换配置文件)
                        if attempt == 2:
                            self.logger.warning(
                                "(Worker) Tier 2 Recovery: Switching Auth Profile..."
                            )
                            try:
                                await self._switch_auth_profile(req_id)
                            except asyncio.CancelledError:
                                raise
                            except Exception as switch_err:
                                self.logger.error(
                                    f"(Worker) Tier 2 Profile Switch failed: {switch_err}"
                                )
                                if "exhausted" in str(switch_err).lower():
                                    self.logger.critical(
                                        "(Worker) Auth profiles exhausted."
                                    )
                                    raise
                            continue

                # --- End Fast-Fail Tiered Error Recovery Logic ---

            # 6. Cleanup / Post-processing (Clear Stream Queue & Chat History)
            await self._cleanup_after_processing(req_id)

            self.logger.info("(Worker) Released processing lock.")

        # Update state for next iteration
        self.was_last_request_streaming = is_streaming_request
        self.last_request_completion_time = time.time()
        if self.request_queue:
            self.request_queue.task_done()

    async def _execute_request_logic(
        self,
        req_id: str,
        request_data: ChatCompletionRequest,
        http_request: Request,
        result_future: Future[Union[StreamingResponse, JSONResponse]],
    ) -> None:
        """Execute the actual request processing logic.

        In multi-session mode, this acquires a session from the pool and uses
        its lock. In single-session mode (default), uses the global page instance.
        """
        # 确保日志上下文已设置
        set_request_id(req_id)

        from api_utils.server_state import state

        # Check if multi-session mode is enabled
        session = None
        if state.session_manager:
            try:
                session = await state.session_manager.get_session()
                self.logger.info(
                    f"(Worker) Using session: {session.session_id}"
                )
            except RuntimeError as e:
                self.logger.error(f"(Worker) Failed to get session: {e}")
                if not result_future.done():
                    result_future.set_exception(
                        server_error(req_id, f"No available sessions: {e}")
                    )
                raise

        try:
            # If using multi-session mode, acquire the session's lock
            if session:
                async with session.lock:
                    await self._execute_with_page(
                        req_id, request_data, http_request, result_future, session.page
                    )
            else:
                # Single-session mode: use global page instance
                await self._execute_with_page(
                    req_id, request_data, http_request, result_future, state.page_instance
                )

        except asyncio.CancelledError:
            self.logger.info("(Worker) Execution cancelled.")
            raise
        except Exception as process_err:
            self.logger.error(f"(Worker) Execution error: {process_err}")
            if not result_future.done():
                result_future.set_exception(
                    server_error(req_id, f"Request processing error: {process_err}")
                )
            # Mark session as failed in multi-session mode
            if session and state.session_manager:
                state.session_manager.mark_session_failed(session)
            # 重新抛出异常以触发重试机制
            raise

    async def _execute_with_page(
        self,
        req_id: str,
        request_data: ChatCompletionRequest,
        http_request: Request,
        result_future: Future[Union[StreamingResponse, JSONResponse]],
        page: Any,  # AsyncPage but avoiding import cycle
    ) -> None:
        """Execute request processing with a specific page instance."""
        from api_utils import (
            _process_request_refactored,  # pyright: ignore[reportPrivateUsage]
        )

        # Store these for cleanup usage if needed
        self.current_submit_btn_loc = None
        self.current_client_disco_checker = None
        self.current_completion_event = None
        self.current_req_id = req_id

        returned_value: Optional[
            Tuple[Optional[Event], Locator, Callable[[str], bool]]
        ] = await _process_request_refactored(
            req_id, request_data, http_request, result_future
        )

        # Initialize variables that will be set from tuple unpacking
        completion_event: Optional[Event] = None
        submit_btn_loc: Optional[Locator] = None
        client_disco_checker: Optional[Callable[[str], bool]] = None
        current_request_was_streaming = False

        if returned_value is not None:
            # Always expect 3-tuple: (Optional[Event], Locator, Callable)
            completion_event, submit_btn_loc, client_disco_checker = returned_value

            if completion_event is not None:
                current_request_was_streaming = True
                self.logger.info("(Worker) Stream info received.")
            else:
                self.logger.info(
                    "(Worker) Tuple received but completion_event is None."
                )
        else:
            self.logger.info("(Worker) Non-stream completion (None).")

        # Store for cleanup
        self.current_submit_btn_loc = submit_btn_loc
        self.current_client_disco_checker = client_disco_checker
        self.current_completion_event = completion_event

        # Initialize stream_state for monitoring
        stream_state: Optional[Dict[str, Any]] = None

        await self._monitor_completion(
            req_id,
            http_request,
            result_future,
            completion_event,
            submit_btn_loc,
            client_disco_checker,
            current_request_was_streaming,
            stream_state,
        )

    async def _monitor_completion(
        self,
        req_id: str,
        http_request: Request,
        result_future: Future[Union[StreamingResponse, JSONResponse]],
        completion_event: Optional[Event],
        submit_btn_loc: Optional[Locator],
        client_disco_checker: Optional[Callable[[str], bool]],
        current_request_was_streaming: bool,
        stream_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Monitor for completion and handle disconnects.

        Args:
            stream_state: 可选的流状态字典，包含 'has_content' 键表示是否收到内容。
                          如果流完成但没有内容，将抛出异常触发重试机制。
        """
        # 确保日志上下文已设置
        set_request_id(req_id)

        from api_utils.client_connection import (
            enhanced_disconnect_monitor,
            non_streaming_disconnect_monitor,
        )

        try:
            from server import RESPONSE_COMPLETION_TIMEOUT
        except ImportError:
            from config import RESPONSE_COMPLETION_TIMEOUT

        disconnect_monitor_task: Optional[Task[bool]] = None
        try:
            if completion_event:
                self.logger.info("(Worker) Waiting for stream completion...")
                disconnect_monitor_task = asyncio.create_task(
                    enhanced_disconnect_monitor(
                        req_id, http_request, completion_event, self.logger
                    )
                )

                await asyncio.wait_for(
                    completion_event.wait(),
                    timeout=RESPONSE_COMPLETION_TIMEOUT / 1000 + 60,
                )
            else:
                self.logger.info("(Worker) Waiting for non-stream completion...")
                disconnect_monitor_task = asyncio.create_task(
                    non_streaming_disconnect_monitor(
                        req_id, http_request, result_future, self.logger
                    )
                )

                await asyncio.wait_for(
                    asyncio.shield(result_future),
                    timeout=RESPONSE_COMPLETION_TIMEOUT / 1000 + 60,
                )

            # Check if client disconnected early
            client_disconnected_early = False
            if disconnect_monitor_task.done():
                try:
                    client_disconnected_early = disconnect_monitor_task.result()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

            self.logger.info(
                f"(Worker) Processing complete. Early disconnect: {client_disconnected_early}"
            )

            # 检查流响应是否包含内容 - 如果为空响应则抛出异常触发重试
            if (
                stream_state is not None
                and completion_event is not None
                and not client_disconnected_early
            ):
                has_content = stream_state.get(
                    "has_content", True
                )  # 默认为 True 避免误报
                if not has_content:
                    self.logger.warning(
                        "(Worker) 检测到空响应 (stream_state.has_content=False)，"
                        "可能是配额用尽或其他错误，抛出异常触发重试机制"
                    )
                    raise RuntimeError(
                        "流响应完成但未收到任何内容 (Empty response - possible quota exceeded)"
                    )

            if (
                not client_disconnected_early
                and submit_btn_loc
                and client_disco_checker
                and completion_event
            ):
                await self._handle_post_stream_button(
                    req_id, submit_btn_loc, client_disco_checker, completion_event
                )

        except asyncio.TimeoutError:
            self.logger.warning("(Worker) Processing timed out.")
            if not result_future.done():
                result_future.set_exception(
                    processing_timeout(
                        req_id, "Processing timed out waiting for completion."
                    )
                )
        except asyncio.CancelledError:
            self.logger.info("(Worker) Completion monitoring cancelled.")
            raise
        except Exception as ev_wait_err:
            self.logger.error(f"(Worker) Error waiting for completion: {ev_wait_err}")
            if not result_future.done():
                result_future.set_exception(
                    server_error(req_id, f"Error waiting for completion: {ev_wait_err}")
                )
            # 重新抛出异常以触发重试机制 (特别是空响应错误)
            raise
        finally:
            if disconnect_monitor_task and not disconnect_monitor_task.done():
                disconnect_monitor_task.cancel()
                try:
                    await disconnect_monitor_task
                except asyncio.CancelledError:
                    pass

    async def _handle_post_stream_button(
        self,
        req_id: str,
        submit_btn_loc: Locator,
        client_disco_checker: Callable[[str], bool],
        completion_event: Event,
    ) -> None:
        """Handle the submit button state after streaming."""
        # 确保日志上下文已设置
        set_request_id(req_id)

        self.logger.info("(Worker) Handling post-stream button state...")
        try:
            from browser_utils.page_controller import PageController
            from server import page_instance

            if page_instance:
                page_controller = PageController(page_instance, self.logger, req_id)
                await page_controller.ensure_generation_stopped(client_disco_checker)
            else:
                self.logger.warning(
                    "(Worker) page_instance is None during button handling"
                )

        except asyncio.CancelledError:
            self.logger.info("(Worker) Post-stream button handling cancelled.")
            raise
        except Exception as e_ensure_stop:
            self.logger.warning(f"Post-stream button handling error: {e_ensure_stop}")
            # Use comprehensive snapshot for better debugging
            import os

            from browser_utils.debug_utils import (
                save_comprehensive_snapshot,
            )
            from config import PROMPT_TEXTAREA_SELECTOR
            from server import page_instance

            if page_instance:
                await save_comprehensive_snapshot(
                    page=page_instance,
                    error_name="stream_post_submit_button_handling_timeout",
                    req_id=req_id,
                    error_stage="流式响应后按钮状态处理",
                    additional_context={
                        "headless_mode": os.environ.get("HEADLESS", "true").lower()
                        == "true",
                        "completion_event_set": completion_event.is_set()
                        if completion_event
                        else None,
                        "error_type": type(e_ensure_stop).__name__,
                        "error_message": str(e_ensure_stop),
                    },
                    locators={
                        "submit_button": submit_btn_loc,
                        "input_field": page_instance.locator(PROMPT_TEXTAREA_SELECTOR),
                    },
                    error_exception=e_ensure_stop,
                )

    async def _cleanup_after_processing(self, req_id: str):
        """Clean up stream queue and chat history."""
        # 确保日志上下文已设置
        set_request_id(req_id)

        try:
            from api_utils import clear_stream_queue

            await clear_stream_queue()

            # Clear chat history if we have the necessary context
            if getattr(self, "current_submit_btn_loc", None) and getattr(
                self, "current_client_disco_checker", None
            ):
                from server import is_page_ready, page_instance

                if (
                    page_instance
                    and is_page_ready
                    and self.current_client_disco_checker
                ):
                    from browser_utils.page_controller import PageController

                    page_controller = PageController(page_instance, self.logger, req_id)

                    self.logger.info("(Worker) Clearing chat history...")
                    await page_controller.clear_chat_history(
                        self.current_client_disco_checker
                    )
                    self.logger.info("(Worker) Chat history cleared.")
        except asyncio.CancelledError:
            self.logger.info("(Worker) Cleanup cancelled.")
            raise
        except Exception as clear_err:
            self.logger.error(f"(Worker) Cleanup error: {clear_err}", exc_info=True)

    async def _refresh_page(self, req_id: str) -> None:
        """Tier 1 Recovery: 快速页面刷新 (~2-3秒)。"""
        # 确保日志上下文已设置
        set_request_id(req_id)

        from api_utils.server_state import state

        if state.page_instance is None:
            raise RuntimeError("page_instance is missing")

        page = state.page_instance
        self.logger.info("(Recovery) 执行页面刷新...")

        try:
            # 快速刷新页面
            await page.reload(wait_until="domcontentloaded", timeout=10000)

            # 等待关键元素可用 (短超时)
            from config.selectors import PROMPT_TEXTAREA_SELECTOR

            await page.wait_for_selector(PROMPT_TEXTAREA_SELECTOR, timeout=5000)

            self.logger.info("(Recovery) 页面刷新完成")
        except asyncio.CancelledError:
            self.logger.info("(Recovery) 页面刷新被取消")
            raise
        except Exception as e:
            self.logger.error(f"(Recovery) 页面刷新失败: {e}")
            raise

    async def _switch_auth_profile(self, req_id: str) -> None:
        """Tier 2 Recovery: 完全重新初始化浏览器连接以避免状态保留。"""
        # 确保日志上下文已设置
        set_request_id(req_id)

        from api_utils.auth_manager import auth_manager
        from browser_utils.initialization.core import (
            close_page_logic,
            enable_temporary_chat_mode,
            initialize_page_logic,
        )
        from browser_utils.model_management import (
            _handle_initial_model_state_and_storage,  # pyright: ignore[reportPrivateUsage]
        )
        from config import get_environment_variable

        # 标记当前配置文件为失败
        auth_manager.mark_profile_failed()

        # 获取下一个配置文件
        next_profile = await auth_manager.get_next_profile()
        self.logger.info(f"(Recovery) 切换到配置文件: {next_profile}")

        # 1. 关闭现有页面
        await close_page_logic()

        # 2. 关闭浏览器连接以获取全新状态
        from api_utils.server_state import state

        if state.browser_instance and state.browser_instance.is_connected():
            await state.browser_instance.close()
            state.is_browser_connected = False
            self.logger.info("(Recovery) 浏览器连接已关闭")

        # 3. 重新连接到 Camoufox
        ws_endpoint = get_environment_variable("CAMOUFOX_WS_ENDPOINT")
        if not ws_endpoint:
            raise RuntimeError("CAMOUFOX_WS_ENDPOINT not available for reconnection")

        if not state.playwright_manager:
            raise RuntimeError("Playwright manager not available")

        self.logger.info("(Recovery) 重新连接到浏览器...")
        state.browser_instance = await state.playwright_manager.firefox.connect(
            ws_endpoint, timeout=30000
        )
        state.is_browser_connected = True
        self.logger.info(f"(Recovery) 已连接: {state.browser_instance.version}")

        # 4. 使用新配置文件初始化页面
        state.page_instance, state.is_page_ready = await initialize_page_logic(
            state.browser_instance,
            storage_state_path=next_profile,
        )

        # 5. 处理初始模型状态和存储
        if state.is_page_ready and state.page_instance:
            await _handle_initial_model_state_and_storage(state.page_instance)

            # 6. 启用临时聊天模式
            await enable_temporary_chat_mode(state.page_instance)

            self.logger.info("(Recovery) 配置文件切换完成 (浏览器已完全重新初始化)")
        else:
            raise RuntimeError("(Recovery) 页面初始化失败，无法完成配置文件切换")


async def queue_worker() -> None:
    """Main queue worker entry point."""
    manager = QueueManager()
    manager.initialize_globals()

    logger = manager.logger
    logger.info("--- Queue Worker Started ---")

    while True:
        try:
            await manager.check_queue_disconnects()

            request_item = await manager.get_next_request()
            if request_item:
                await manager.process_request(request_item)

        except asyncio.CancelledError:
            logger.info("--- Queue Worker Cancelled ---")
            break
        except Exception as e:
            logger.error(f"(Worker) Unexpected error in main loop: {e}", exc_info=True)
            await asyncio.sleep(1)  # Prevent tight loop on error

    logger.info("--- Queue Worker Stopped ---")
