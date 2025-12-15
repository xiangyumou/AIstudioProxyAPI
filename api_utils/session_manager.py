"""
Session Manager Module

Provides multi-session pooling for parallel request processing.
Each session corresponds to an independent browser context with its own
authentication profile, enabling concurrent request handling and fault tolerance.

Usage:
    from api_utils.session_manager import SessionManager

    manager = SessionManager()
    await manager.initialize(browser, auth_dir)
    session = await manager.get_session()
    async with session.lock:
        # Use session.page for browser operations
        ...
"""

import asyncio
import glob
import logging
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import BrowserContext as AsyncBrowserContext
    from playwright.async_api import Page as AsyncPage

logger = logging.getLogger("session_manager")


class Session:
    """
    Encapsulates a single browser session.

    Each session has its own browser context, page, and lock to ensure
    thread-safe access during request processing.

    Attributes:
        profile_path: Path to the authentication JSON file.
        context: The Playwright browser context.
        page: The Playwright page instance.
        lock: Asyncio lock for exclusive session access.
        is_ready: Whether the session is initialized and ready.
        current_model_id: The currently selected model for this session.
    """

    def __init__(self, profile_path: str) -> None:
        """Initialize session with authentication profile path."""
        self.profile_path = profile_path
        self.context: Optional["AsyncBrowserContext"] = None
        self.page: Optional["AsyncPage"] = None
        self.lock = asyncio.Lock()
        self.is_ready = False
        self.current_model_id: Optional[str] = None
        self._session_id = os.path.basename(profile_path).replace(".json", "")

    @property
    def session_id(self) -> str:
        """Readable session identifier based on profile filename."""
        return self._session_id

    async def initialize(self, browser: "AsyncBrowser") -> None:
        """
        Initialize the browser context and page for this session.

        Args:
            browser: The Playwright browser instance to create context from.

        Raises:
            RuntimeError: If initialization fails.
        """
        from browser_utils.initialization.core import initialize_page_logic

        logger.info(f"[Session:{self.session_id}] Initializing...")

        try:
            # initialize_page_logic creates context internally and returns (page, is_ready)
            self.page, self.is_ready = await initialize_page_logic(
                browser, storage_state_path=self.profile_path
            )

            if self.page and self.is_ready:
                # Get the context from the page
                self.context = self.page.context
                logger.info(f"[Session:{self.session_id}] Initialized successfully")
            else:
                raise RuntimeError(
                    f"Page initialization returned is_ready=False for {self.profile_path}"
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[Session:{self.session_id}] Initialization failed: {e}")
            self.is_ready = False
            raise RuntimeError(
                f"Failed to initialize session {self.session_id}: {e}"
            ) from e

    async def close(self) -> None:
        """Close the session's browser context and page."""
        logger.info(f"[Session:{self.session_id}] Closing...")

        if self.page and not self.page.is_closed():
            try:
                await self.page.close()
            except Exception as e:
                logger.warning(f"[Session:{self.session_id}] Error closing page: {e}")

        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                logger.warning(
                    f"[Session:{self.session_id}] Error closing context: {e}"
                )

        self.page = None
        self.context = None
        self.is_ready = False
        logger.info(f"[Session:{self.session_id}] Closed")


class SessionManager:
    """
    Manages a pool of browser sessions for concurrent request processing.

    Uses round-robin scheduling to distribute requests across available sessions.
    Each session corresponds to a separate authentication profile, allowing
    parallel processing and automatic failover on quota exhaustion.

    Attributes:
        sessions: List of managed Session instances.
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        self.sessions: List[Session] = []
        self._index = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(
        self,
        browser: "AsyncBrowser",
        auth_dir: str,
        max_sessions: int = 0,
    ) -> None:
        """
        Initialize sessions from authentication files in the specified directory.

        Args:
            browser: The Playwright browser instance.
            auth_dir: Directory containing authentication JSON files.
            max_sessions: Maximum number of sessions to create (0 = no limit).

        Raises:
            RuntimeError: If no valid authentication files found or all fail.
        """
        if self._initialized:
            logger.warning("SessionManager already initialized, skipping")
            return

        logger.info(f"Initializing SessionManager from: {auth_dir}")

        # Find all JSON files in the auth directory
        if not os.path.exists(auth_dir):
            raise RuntimeError(f"Authentication directory not found: {auth_dir}")

        pattern = os.path.join(auth_dir, "*.json")
        profile_paths = sorted(glob.glob(pattern))

        if not profile_paths:
            raise RuntimeError(f"No authentication files found in: {auth_dir}")

        # Apply max_sessions limit if specified
        if max_sessions > 0:
            profile_paths = profile_paths[:max_sessions]

        logger.info(f"Found {len(profile_paths)} authentication profile(s)")

        # Initialize sessions
        successful = 0
        for profile_path in profile_paths:
            session = Session(profile_path)
            try:
                await session.initialize(browser)
                self.sessions.append(session)
                successful += 1
                logger.info(
                    f"Session {successful}/{len(profile_paths)} initialized: "
                    f"{session.session_id}"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Failed to initialize session from {profile_path}: {e}")
                # Continue with remaining profiles

        if not self.sessions:
            raise RuntimeError("All session initializations failed")

        self._initialized = True
        logger.info(f"SessionManager initialized with {len(self.sessions)} session(s)")

    async def get_session(self) -> Session:
        """
        Get the next available session using round-robin scheduling.

        Returns:
            A ready Session instance.

        Raises:
            RuntimeError: If no sessions are available or all are not ready.
        """
        async with self._lock:
            if not self.sessions:
                raise RuntimeError("No sessions available")

            # Find a ready session using round-robin
            attempts = 0
            while attempts < len(self.sessions):
                session = self.sessions[self._index]
                self._index = (self._index + 1) % len(self.sessions)

                if session.is_ready:
                    logger.debug(f"Selected session: {session.session_id}")
                    return session

                attempts += 1

            raise RuntimeError("All sessions are not ready")

    def get_default_session(self) -> Optional[Session]:
        """
        Get the first session (for backward compatibility).

        Returns:
            The first session if available, None otherwise.
        """
        if self.sessions:
            return self.sessions[0]
        return None

    async def close_all(self) -> None:
        """Close all managed sessions."""
        logger.info(f"Closing all {len(self.sessions)} session(s)...")

        for session in self.sessions:
            try:
                await session.close()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error closing session {session.session_id}: {e}")

        self.sessions.clear()
        self._index = 0
        self._initialized = False
        logger.info("All sessions closed")

    def mark_session_failed(self, session: Session) -> None:
        """
        Mark a session as failed (not ready).

        Args:
            session: The session to mark as failed.
        """
        session.is_ready = False
        logger.warning(f"Session marked as failed: {session.session_id}")

    async def recover_session(self, session: Session, browser: "AsyncBrowser") -> bool:
        """
        Attempt to recover a failed session.

        Args:
            session: The session to recover.
            browser: The browser instance for reinitialization.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        logger.info(f"Attempting to recover session: {session.session_id}")

        try:
            await session.close()
            await session.initialize(browser)
            return session.is_ready
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Session recovery failed: {e}")
            return False

    @property
    def active_session_count(self) -> int:
        """Count of sessions that are ready."""
        return sum(1 for s in self.sessions if s.is_ready)

    @property
    def total_session_count(self) -> int:
        """Total number of managed sessions."""
        return len(self.sessions)
