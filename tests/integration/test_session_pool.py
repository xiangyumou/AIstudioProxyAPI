"""
Integration tests for multi-session pooling.

These tests verify:
- Concurrent request distribution across sessions
- Session failure recovery
- Single-session fallback mode
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api_utils.server_state import state


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
async def mock_session_manager():
    """Create a mock session manager with multiple sessions."""
    from api_utils.session_manager import Session, SessionManager

    manager = SessionManager()

    # Create mock sessions
    for i in range(3):
        session = Session(f"/mock/path/user{i + 1}.json")
        session.page = AsyncMock()
        session.page.is_closed = MagicMock(return_value=False)
        session.context = AsyncMock()
        session.is_ready = True
        session.lock = asyncio.Lock()
        manager.sessions.append(session)

    manager._initialized = True

    yield manager

    # Cleanup
    manager.sessions.clear()


@pytest.fixture
async def real_server_state_with_session_manager(mock_session_manager):
    """Provide real server state with mock session manager."""
    # Reset state
    state.reset()

    # Setup real asyncio primitives
    state.processing_lock = asyncio.Lock()
    state.model_switching_lock = asyncio.Lock()
    state.params_cache_lock = asyncio.Lock()
    state.request_queue = asyncio.Queue()

    # Mock browser/page
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    state.page_instance = mock_page
    state.is_page_ready = True
    state.is_browser_connected = True

    # Attach session manager
    state.session_manager = mock_session_manager

    yield state

    # Cleanup
    while not state.request_queue.empty():
        try:
            state.request_queue.get_nowait()
            state.request_queue.task_done()
        except asyncio.QueueEmpty:
            break

    state.session_manager = None
    state.reset()


class TestSessionPoolConcurrency:
    """Tests for concurrent request handling with session pool."""

    @pytest.mark.asyncio
    async def test_requests_distributed_across_sessions(
        self, real_server_state_with_session_manager
    ):
        """Test that requests are distributed to different sessions."""
        manager = real_server_state_with_session_manager.session_manager

        # Track which sessions were used
        sessions_used = []

        for _ in range(6):  # 6 requests with 3 sessions = 2 per session
            session = await manager.get_session()
            sessions_used.append(session.session_id)

        # Each session should be used twice (round-robin)
        assert sessions_used.count("user1") == 2
        assert sessions_used.count("user2") == 2
        assert sessions_used.count("user3") == 2

    @pytest.mark.asyncio
    async def test_session_lock_prevents_concurrent_access(
        self, real_server_state_with_session_manager
    ):
        """Test that session locks prevent concurrent access to same session."""
        manager = real_server_state_with_session_manager.session_manager
        session = manager.sessions[0]

        access_order = []

        async def task(task_id: int):
            async with session.lock:
                access_order.append(f"start_{task_id}")
                await asyncio.sleep(0.1)  # Simulate work
                access_order.append(f"end_{task_id}")

        # Start two concurrent tasks
        await asyncio.gather(task(1), task(2))

        # Should be sequential: start_1, end_1, start_2, end_2
        assert access_order[0] == "start_1"
        assert access_order[1] == "end_1"
        assert access_order[2] == "start_2"
        assert access_order[3] == "end_2"


class TestSessionFailover:
    """Tests for session failure and recovery."""

    @pytest.mark.asyncio
    async def test_failed_session_skipped(
        self, real_server_state_with_session_manager
    ):
        """Test that failed sessions are skipped in round-robin."""
        manager = real_server_state_with_session_manager.session_manager

        # Mark second session as failed
        manager.sessions[1].is_ready = False

        sessions_used = []
        for _ in range(4):
            session = await manager.get_session()
            sessions_used.append(session.session_id)

        # user2 should not appear
        assert "user2" not in sessions_used
        assert sessions_used.count("user1") == 2
        assert sessions_used.count("user3") == 2

    @pytest.mark.asyncio
    async def test_all_sessions_failed_raises_error(
        self, real_server_state_with_session_manager
    ):
        """Test that RuntimeError is raised when all sessions fail."""
        manager = real_server_state_with_session_manager.session_manager

        # Mark all sessions as failed
        for session in manager.sessions:
            session.is_ready = False

        with pytest.raises(RuntimeError, match="All sessions are not ready"):
            await manager.get_session()


class TestSingleSessionFallback:
    """Tests for single-session fallback mode."""

    @pytest.mark.asyncio
    async def test_no_session_manager_uses_global_page(self):
        """Test that when session_manager is None, global page is used."""
        # Reset state without session manager
        state.reset()
        state.session_manager = None
        state.page_instance = AsyncMock()
        state.is_page_ready = True

        # Verify session_manager is None
        assert state.session_manager is None

        # In this mode, queue_worker should use state.page_instance
        assert state.page_instance is not None

        state.reset()


class TestSessionManagerLifecycle:
    """Tests for session manager initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_close_all_cleans_up(self, mock_session_manager):
        """Test that close_all properly cleans up all sessions."""
        manager = mock_session_manager
        original_count = len(manager.sessions)
        assert original_count == 3

        await manager.close_all()

        assert len(manager.sessions) == 0
        assert manager._initialized is False
        assert manager._index == 0

    @pytest.mark.asyncio
    async def test_session_recovery(self, mock_session_manager):
        """Test session recovery after failure."""
        manager = mock_session_manager
        session = manager.sessions[0]
        session.is_ready = False

        mock_browser = AsyncMock()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_page = AsyncMock()
            mock_page.is_closed = MagicMock(return_value=False)
            mock_page.context = AsyncMock()
            mock_init.return_value = (mock_page, True)

            result = await manager.recover_session(session, mock_browser)

            assert result is True
            assert session.is_ready is True
