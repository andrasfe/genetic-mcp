"""Unit tests for client-generated ideas functionality."""

import asyncio
from asyncio import TimeoutError
from unittest.mock import AsyncMock, Mock

import pytest

from genetic_mcp.models import (
    EvolutionMode,
    GenerationRequest,
)
from genetic_mcp.session_manager import SessionManager


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.embed = AsyncMock(return_value=[0.1] * 768)  # Mock embedding
    client.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])  # Mock batch embedding
    client.generate = AsyncMock(return_value="Generated content")
    client.get_available_models = Mock(return_value=["model1", "model2"])
    return client


@pytest.fixture
async def session_manager(mock_llm_client):
    """Create a session manager with mock LLM client."""
    manager = SessionManager(mock_llm_client)
    await manager.start()
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_create_client_generated_session(session_manager):
    """Test creating a client-generated session."""
    request = GenerationRequest(
        prompt="Test prompt",
        mode=EvolutionMode.ITERATIVE,
        population_size=5,
        client_generated=True,
    )

    session = await session_manager.create_session("test_client", request)

    assert session.client_generated is True
    assert session.prompt == "Test prompt"
    assert session.parameters.population_size == 5
    assert not hasattr(session, '_worker_pool') or session._worker_pool is None
    assert hasattr(session, '_fitness_evaluator')
    assert hasattr(session, '_genetic_algorithm')


@pytest.mark.asyncio
async def test_inject_ideas(session_manager):
    """Test injecting ideas into a client-generated session."""
    # Create client-generated session
    request = GenerationRequest(
        prompt="Test prompt",
        client_generated=True,
        population_size=3,
    )
    session = await session_manager.create_session("test_client", request)

    # Inject ideas
    ideas = ["Idea 1", "Idea 2", "Idea 3"]
    injected = await session_manager.inject_ideas(session, ideas, generation=0)

    assert len(injected) == 3
    assert len(session.ideas) == 3
    assert session.ideas_per_generation_received[0] == 3

    # Check injected ideas
    for i, idea in enumerate(injected):
        assert idea.content == ideas[i]
        assert idea.generation == 0
        assert idea.metadata["source"] == "client"
        assert idea.metadata["injection_index"] == i


@pytest.mark.asyncio
async def test_inject_ideas_validation(session_manager):
    """Test validation when injecting ideas."""
    # Test 1: Non-client-generated session
    request = GenerationRequest(
        prompt="Test prompt",
        client_generated=False,  # Not client-generated
    )
    session = await session_manager.create_session("test_client", request)

    with pytest.raises(ValueError, match="not configured for client-generated ideas"):
        await session_manager.inject_ideas(session, ["idea"], generation=0)

    # Test 2: Completed session
    request2 = GenerationRequest(
        prompt="Test prompt",
        client_generated=True,
    )
    session2 = await session_manager.create_session("test_client", request2)
    session2.status = "completed"

    with pytest.raises(ValueError, match="Cannot inject ideas into session with status 'completed'"):
        await session_manager.inject_ideas(session2, ["idea"], generation=0)


@pytest.mark.asyncio
async def test_wait_for_client_ideas(session_manager):
    """Test waiting for client ideas."""
    # Create session
    request = GenerationRequest(
        prompt="Test prompt",
        client_generated=True,
        population_size=2,
    )
    session = await session_manager.create_session("test_client", request)

    # Start waiting task
    wait_task = asyncio.create_task(
        session_manager._wait_for_client_ideas(
            session, generation=0, expected_count=2, timeout_seconds=2
        )
    )

    # Inject ideas after a short delay
    await asyncio.sleep(0.1)
    await session_manager.inject_ideas(session, ["Idea 1", "Idea 2"], generation=0)

    # Should complete successfully
    ideas = await wait_task
    assert len(ideas) == 2
    assert all(idea.generation == 0 for idea in ideas)


@pytest.mark.asyncio
async def test_wait_for_client_ideas_timeout(session_manager):
    """Test timeout when waiting for client ideas."""
    # Create session
    request = GenerationRequest(
        prompt="Test prompt",
        client_generated=True,
        population_size=5,
    )
    session = await session_manager.create_session("test_client", request)

    # Try to wait with short timeout
    with pytest.raises(TimeoutError, match="Timeout waiting for client ideas"):
        await session_manager._wait_for_client_ideas(
            session, generation=0, expected_count=5, timeout_seconds=0.5
        )


@pytest.mark.asyncio
async def test_run_generation_client_mode(session_manager):
    """Test running generation in client-generated mode."""
    # Create session
    request = GenerationRequest(
        prompt="Test prompt",
        mode=EvolutionMode.SINGLE_PASS,
        client_generated=True,
        population_size=3,
        top_k=2,
    )
    session = await session_manager.create_session("test_client", request)

    # Start generation in background
    gen_task = asyncio.create_task(
        session_manager.run_generation(session, top_k=2)
    )

    # Wait for it to start
    await asyncio.sleep(0.1)

    # Inject ideas
    ideas = ["Idea A", "Idea B", "Idea C"]
    await session_manager.inject_ideas(session, ideas, generation=0)

    # Wait for completion
    result = await gen_task

    assert result.total_ideas_generated == 3
    assert result.generations_completed == 1
    assert len(result.top_ideas) == 2
    assert session.status == "completed"


@pytest.mark.asyncio
async def test_multiple_generations_client_mode(session_manager):
    """Test multiple generations in client-generated mode."""
    # Create session
    request = GenerationRequest(
        prompt="Test prompt",
        mode=EvolutionMode.ITERATIVE,
        client_generated=True,
        population_size=2,
        generations=2,
        top_k=1,
    )
    session = await session_manager.create_session("test_client", request)

    # Start generation in background
    gen_task = asyncio.create_task(
        session_manager.run_generation(session, top_k=1)
    )

    # Wait and inject generation 0
    await asyncio.sleep(0.1)
    await session_manager.inject_ideas(session, ["Gen0-A", "Gen0-B"], generation=0)

    # Wait and inject generation 1
    await asyncio.sleep(0.5)
    await session_manager.inject_ideas(session, ["Gen1-A", "Gen1-B"], generation=1)

    # Wait for completion
    result = await gen_task

    assert result.total_ideas_generated == 6  # 2 + 2 + 2 (elites)
    assert result.generations_completed == 2
    assert session.ideas_per_generation_received == {0: 2, 1: 2}
