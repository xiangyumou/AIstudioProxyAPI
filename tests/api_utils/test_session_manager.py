"""
Unit tests for session_manager module.

Tests cover:
- Session initialization and lifecycle
- SessionManager pool initialization
- Round-robin session scheduling
- Session failure marking and recovery
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api_utils.session_manager import Session, SessionManager


# --- Fixtures ---


@pytest.fixture
def mock_browser():
    """Create a mock browser instance."""
    browser = AsyncMock()
    browser.new_context = AsyncMock()
    return browser


@pytest.fixture
def mock_page():
    """Create a mock page instance."""
    page = AsyncMock()
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()
    page.context = AsyncMock()
    page.context.close = AsyncMock()
    return page


@pytest.fixture
def mock_auth_dir(tmp_path):
    """Create a temporary auth directory with test JSON files."""
    auth_dir = tmp_path / "saved"
    auth_dir.mkdir()

    # Create test auth files
    for i in range(3):
        auth_file = auth_dir / f"user{i + 1}.json"
        auth_file.write_text('{"cookies": []}', encoding="utf-8")

    return str(auth_dir)


@pytest.fixture
def session():
    """Create a Session instance with a mock profile path."""
    return Session("/path/to/auth/user1.json")


# --- Session Tests ---


class TestSession:
    """Tests for the Session class."""

    def test_init(self, session):
        """Test Session initialization."""
        assert session.profile_path == "/path/to/auth/user1.json"
        assert session.session_id == "user1"
        assert session.context is None
        assert session.page is None
        assert session.is_ready is False
        assert session.current_model_id is None
        assert isinstance(session.lock, asyncio.Lock)

    def test_session_id_extraction(self):
        """Test session_id is extracted from filename."""
        session = Session("/some/path/my-account.json")
        assert session.session_id == "my-account"

        session2 = Session("C:\\Users\\test\\auth.json")
        assert session2.session_id == "auth"

    @pytest.mark.asyncio
    async def test_initialize_success(self, session, mock_browser, mock_page):
        """Test successful session initialization."""
        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)

            await session.initialize(mock_browser)

            mock_init.assert_called_once_with(
                mock_browser, storage_state_path=session.profile_path
            )
            assert session.page == mock_page
            assert session.is_ready is True
            assert session.context == mock_page.context

    @pytest.mark.asyncio
    async def test_initialize_failure(self, session, mock_browser, mock_page):
        """Test session initialization failure."""
        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, False)

            with pytest.raises(RuntimeError, match="is_ready=False"):
                await session.initialize(mock_browser)

            assert session.is_ready is False

    @pytest.mark.asyncio
    async def test_initialize_exception(self, session, mock_browser):
        """Test session initialization with exception."""
        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.side_effect = Exception("Connection failed")

            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await session.initialize(mock_browser)

            assert session.is_ready is False

    @pytest.mark.asyncio
    async def test_close(self, session, mock_page):
        """Test session close."""
        session.page = mock_page
        session.context = mock_page.context
        session.is_ready = True

        await session.close()

        mock_page.close.assert_called_once()
        mock_page.context.close.assert_called_once()
        assert session.page is None
        assert session.context is None
        assert session.is_ready is False

    @pytest.mark.asyncio
    async def test_close_already_closed(self, session, mock_page):
        """Test closing an already closed session."""
        mock_page.is_closed = MagicMock(return_value=True)
        session.page = mock_page
        session.context = mock_page.context

        await session.close()

        mock_page.close.assert_not_called()
        mock_page.context.close.assert_called_once()


# --- SessionManager Tests ---


class TestSessionManager:
    """Tests for the SessionManager class."""

    def test_init(self):
        """Test SessionManager initialization."""
        manager = SessionManager()
        assert manager.sessions == []
        assert manager._index == 0
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_browser, mock_page, mock_auth_dir):
        """Test successful SessionManager initialization."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)

            await manager.initialize(mock_browser, mock_auth_dir)

            assert len(manager.sessions) == 3
            assert manager._initialized is True
            assert mock_init.call_count == 3

    @pytest.mark.asyncio
    async def test_initialize_no_directory(self, mock_browser):
        """Test initialization with non-existent directory."""
        manager = SessionManager()

        with pytest.raises(RuntimeError, match="not found"):
            await manager.initialize(mock_browser, "/non/existent/path")

    @pytest.mark.asyncio
    async def test_initialize_no_files(self, mock_browser, tmp_path):
        """Test initialization with empty directory."""
        manager = SessionManager()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError, match="No authentication files"):
            await manager.initialize(mock_browser, str(empty_dir))

    @pytest.mark.asyncio
    async def test_initialize_with_max_sessions(
        self, mock_browser, mock_page, mock_auth_dir
    ):
        """Test initialization with max_sessions limit."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)

            await manager.initialize(mock_browser, mock_auth_dir, max_sessions=2)

            assert len(manager.sessions) == 2
            assert mock_init.call_count == 2

    @pytest.mark.asyncio
    async def test_initialize_partial_failure(
        self, mock_browser, mock_page, mock_auth_dir
    ):
        """Test initialization when some sessions fail."""
        manager = SessionManager()
        call_count = 0

        async def mock_init_logic(browser, storage_state_path=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Auth expired")
            return (mock_page, True)

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            side_effect=mock_init_logic,
        ):
            await manager.initialize(mock_browser, mock_auth_dir)

            # 2 out of 3 should succeed
            assert len(manager.sessions) == 2
            assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_all_fail(self, mock_browser, mock_auth_dir):
        """Test initialization when all sessions fail."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.side_effect = Exception("All failed")

            with pytest.raises(RuntimeError, match="All session initializations failed"):
                await manager.initialize(mock_browser, mock_auth_dir)

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(
        self, mock_browser, mock_page, mock_auth_dir
    ):
        """Test that double initialization is prevented."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)

            await manager.initialize(mock_browser, mock_auth_dir)
            await manager.initialize(mock_browser, mock_auth_dir)  # Should skip

            # Only first initialization should happen
            assert mock_init.call_count == 3  # 3 sessions, not 6

    @pytest.mark.asyncio
    async def test_get_session_round_robin(self, mock_browser, mock_page, mock_auth_dir):
        """Test round-robin session selection."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)
            await manager.initialize(mock_browser, mock_auth_dir)

        # Get sessions in round-robin order
        s1 = await manager.get_session()
        s2 = await manager.get_session()
        s3 = await manager.get_session()
        s4 = await manager.get_session()  # Wraps around

        assert s1 == manager.sessions[0]
        assert s2 == manager.sessions[1]
        assert s3 == manager.sessions[2]
        assert s4 == manager.sessions[0]  # Back to first

    @pytest.mark.asyncio
    async def test_get_session_skips_not_ready(
        self, mock_browser, mock_page, mock_auth_dir
    ):
        """Test that get_session skips non-ready sessions."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)
            await manager.initialize(mock_browser, mock_auth_dir)

        # Mark second session as not ready
        manager.sessions[1].is_ready = False

        s1 = await manager.get_session()
        s2 = await manager.get_session()
        s3 = await manager.get_session()

        assert s1 == manager.sessions[0]
        assert s2 == manager.sessions[2]  # Skipped sessions[1]
        assert s3 == manager.sessions[0]  # Wraps, skips sessions[1] again

    @pytest.mark.asyncio
    async def test_get_session_no_sessions(self):
        """Test get_session with empty manager."""
        manager = SessionManager()

        with pytest.raises(RuntimeError, match="No sessions available"):
            await manager.get_session()

    @pytest.mark.asyncio
    async def test_get_session_all_not_ready(
        self, mock_browser, mock_page, mock_auth_dir
    ):
        """Test get_session when all sessions are not ready."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)
            await manager.initialize(mock_browser, mock_auth_dir)

        # Mark all as not ready
        for s in manager.sessions:
            s.is_ready = False

        with pytest.raises(RuntimeError, match="All sessions are not ready"):
            await manager.get_session()

    def test_get_default_session(self, mock_page):
        """Test get_default_session."""
        manager = SessionManager()
        assert manager.get_default_session() is None

        session = Session("/path/to/auth.json")
        session.page = mock_page
        manager.sessions.append(session)

        assert manager.get_default_session() == session

    @pytest.mark.asyncio
    async def test_close_all(self, mock_browser, mock_page, mock_auth_dir):
        """Test closing all sessions."""
        manager = SessionManager()

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)
            await manager.initialize(mock_browser, mock_auth_dir)

        original_count = len(manager.sessions)
        assert original_count == 3

        await manager.close_all()

        assert manager.sessions == []
        assert manager._index == 0
        assert manager._initialized is False

    def test_mark_session_failed(self, mock_page):
        """Test marking a session as failed."""
        manager = SessionManager()
        session = Session("/path/to/auth.json")
        session.page = mock_page
        session.is_ready = True
        manager.sessions.append(session)

        manager.mark_session_failed(session)

        assert session.is_ready is False

    @pytest.mark.asyncio
    async def test_recover_session(self, mock_browser, mock_page):
        """Test session recovery."""
        manager = SessionManager()
        session = Session("/path/to/auth.json")
        session.page = mock_page
        session.context = mock_page.context
        session.is_ready = False

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = (mock_page, True)

            result = await manager.recover_session(session, mock_browser)

            assert result is True
            assert session.is_ready is True

    @pytest.mark.asyncio
    async def test_recover_session_failure(self, mock_browser, mock_page):
        """Test session recovery failure."""
        manager = SessionManager()
        session = Session("/path/to/auth.json")
        session.page = mock_page
        session.context = mock_page.context
        session.is_ready = False

        with patch(
            "browser_utils.initialization.core.initialize_page_logic",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.side_effect = Exception("Recovery failed")

            result = await manager.recover_session(session, mock_browser)

            assert result is False

    def test_active_session_count(self, mock_page):
        """Test active_session_count property."""
        manager = SessionManager()

        for i in range(3):
            session = Session(f"/path/user{i}.json")
            session.page = mock_page
            session.is_ready = i != 1  # Second one is not ready
            manager.sessions.append(session)

        assert manager.active_session_count == 2
        assert manager.total_session_count == 3
