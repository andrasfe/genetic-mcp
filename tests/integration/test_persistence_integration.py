"""Integration test for persistence functionality in session manager."""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock

from genetic_mcp.models import Session, GenerationRequest, EvolutionMode
from genetic_mcp.session_manager import SessionManager
from genetic_mcp.llm_client import MultiModelClient, OpenAIClient


@pytest.fixture
async def mock_session_manager():
    """Create a mock session manager for testing."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    # Create mock LLM client
    mock_llm_client = MagicMock(spec=MultiModelClient)
    mock_llm_client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    # Create real session manager
    manager = SessionManager(
        llm_client=mock_llm_client,
        persistence_db_path=db_path,
        enable_auto_save=True
    )
    await manager.start()
    
    yield manager
    
    # Cleanup
    await manager.stop()
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
async def test_session(mock_session_manager):
    """Create a test session."""
    request = GenerationRequest(
        prompt="Test persistence integration",
        mode=EvolutionMode.SINGLE_PASS,
        population_size=5,
        client_generated=True  # No LLM calls needed
    )
    
    session = await mock_session_manager.create_session("test-client", request)
    return session


class TestPersistenceIntegration:
    """Test persistence functionality integration."""
    
    @pytest.mark.asyncio
    async def test_save_session_tool(self, mock_session_manager, test_session):
        """Test the save_session functionality through session manager."""
        success = await mock_session_manager.save_session_to_db(
            session_id=test_session.id,
            checkpoint_name="integration_test"
        )
        
        assert success is True
        
        # Verify it was saved by loading it back
        loaded = await mock_session_manager.load_session_from_db(test_session.id)
        assert loaded is not None
        assert loaded.id == test_session.id
    
    @pytest.mark.asyncio
    async def test_save_session_tool_nonexistent(self, mock_session_manager):
        """Test save_session with nonexistent session."""
        success = await mock_session_manager.save_session_to_db(
            session_id="nonexistent-session",
            checkpoint_name="test"
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_load_session_tool(self, mock_session_manager, test_session):
        """Test the load_session functionality."""
        # First save the session
        await mock_session_manager.save_session_to_db(test_session.id)
        
        # Remove from active sessions
        await mock_session_manager.delete_session(test_session.id)
        
        # Load session
        loaded_session = await mock_session_manager.load_session_from_db(test_session.id)
        
        assert loaded_session is not None
        assert loaded_session.id == test_session.id
        assert loaded_session.prompt == test_session.prompt
        assert loaded_session.client_id == test_session.client_id
    
    @pytest.mark.asyncio
    async def test_load_session_tool_nonexistent(self, mock_session_manager):
        """Test load_session with nonexistent session."""
        loaded_session = await mock_session_manager.load_session_from_db("nonexistent-session")
        assert loaded_session is None
    
    @pytest.mark.asyncio
    async def test_resume_session_tool(self, mock_session_manager, test_session):
        """Test the resume_session functionality."""
        # First save the session
        await mock_session_manager.save_session_to_db(test_session.id)
        
        # Change status to completed
        test_session.status = "completed"
        await mock_session_manager.save_session_to_db(test_session.id)
        
        # Resume session
        success = await mock_session_manager.resume_session(test_session.id)
        
        assert success is True
        
        # Verify session is active
        active_session = await mock_session_manager.get_session(test_session.id)
        assert active_session is not None
        assert active_session.status == "active"
    
    @pytest.mark.asyncio
    async def test_resume_session_tool_nonexistent(self, mock_session_manager):
        """Test resume_session with nonexistent session."""
        success = await mock_session_manager.resume_session("nonexistent-session")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_list_saved_sessions_tool(self, mock_session_manager, test_session):
        """Test the list_saved_sessions functionality."""
        # Save the session
        await mock_session_manager.save_session_to_db(test_session.id)
        
        # List sessions
        sessions = await mock_session_manager.list_saved_sessions(limit=10, offset=0)
        
        assert isinstance(sessions, list)
        assert len(sessions) >= 1
        
        # Find our session
        session_found = False
        for session_info in sessions:
            if session_info["id"] == test_session.id:
                assert session_info["client_id"] == test_session.client_id
                assert test_session.prompt in session_info["prompt"]  # May be truncated
                session_found = True
                break
        
        assert session_found, "Test session not found in list"
    
    @pytest.mark.asyncio
    async def test_list_saved_sessions_tool_with_filter(self, mock_session_manager, test_session):
        """Test list_saved_sessions with client filter."""
        await mock_session_manager.save_session_to_db(test_session.id)
        
        # List sessions for specific client
        sessions = await mock_session_manager.list_saved_sessions(
            client_id=test_session.client_id,
            limit=10,
            offset=0
        )
        
        assert len(sessions) >= 1
        assert all(
            session["client_id"] == test_session.client_id 
            for session in sessions
        )
    
    @pytest.mark.asyncio
    async def test_full_persistence_workflow(self, mock_session_manager, test_session):
        """Test complete persistence workflow."""
        # 1. Save session
        save_success = await mock_session_manager.save_session_to_db(test_session.id, "workflow_test")
        assert save_success is True
        
        # 2. Remove from memory
        await mock_session_manager.delete_session(test_session.id)
        
        # 3. List sessions (should find it)
        sessions = await mock_session_manager.list_saved_sessions()
        assert any(s["id"] == test_session.id for s in sessions)
        
        # 4. Load session back
        loaded = await mock_session_manager.load_session_from_db(test_session.id)
        assert loaded is not None
        
        # 5. Resume session
        resume_success = await mock_session_manager.resume_session(test_session.id)
        assert resume_success is True
        
        # 6. Verify session is active
        active_session = await mock_session_manager.get_session(test_session.id)
        assert active_session is not None
        assert active_session.status == "active"