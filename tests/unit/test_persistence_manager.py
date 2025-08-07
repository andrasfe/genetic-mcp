"""Test the persistence manager functionality."""

import asyncio
import contextlib
import os
import tempfile

import pytest

from genetic_mcp.models import (
    EvolutionMode,
    FitnessWeights,
    GeneticParameters,
    Idea,
    Session,
    Worker,
    WorkerStatus,
)
from genetic_mcp.persistence_manager import PersistenceManager


@pytest.fixture
async def persistence_manager():
    """Create a temporary persistence manager for testing."""
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    manager = PersistenceManager(db_path)
    await manager.initialize()

    yield manager

    # Cleanup
    with contextlib.suppress(Exception):
        os.unlink(db_path)


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    session = Session(
        id="test-session-123",
        client_id="test-client",
        prompt="Generate innovative AI application ideas",
        mode=EvolutionMode.ITERATIVE,
        parameters=GeneticParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_count=2
        ),
        fitness_weights=FitnessWeights(
            relevance=0.4,
            novelty=0.3,
            feasibility=0.3
        ),
        status="active",
        current_generation=2,
        adaptive_population_enabled=True,
        adaptive_population_config={
            "min_population": 5,
            "max_population": 20,
            "diversity_threshold": 0.3
        },
        memory_enabled=True,
        parameter_recommendation={"confidence": 0.8, "source": "test"}
    )

    # Add sample ideas
    ideas = [
        Idea(
            id="idea-1",
            content="AI-powered code review assistant",
            generation=0,
            fitness=0.85,
            scores={"relevance": 0.9, "novelty": 0.8, "feasibility": 0.85},
            metadata={"source": "initial"}
        ),
        Idea(
            id="idea-2",
            content="Automated test case generator",
            generation=1,
            parent_ids=["idea-1"],
            fitness=0.92,
            scores={"relevance": 0.95, "novelty": 0.85, "feasibility": 0.95},
            metadata={"source": "crossover"}
        ),
    ]
    session.ideas = ideas

    # Add sample workers
    workers = [
        Worker(
            id="worker-1",
            status=WorkerStatus.WORKING,
            model="gpt-4-turbo",
            completed_tasks=5,
            failed_tasks=1
        ),
        Worker(
            id="worker-2",
            status=WorkerStatus.IDLE,
            model="claude-3-opus",
            completed_tasks=3,
            failed_tasks=0
        )
    ]
    session.workers = workers

    return session


class TestPersistenceManager:
    """Test persistence manager functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, persistence_manager):
        """Test persistence manager initialization."""
        # Should be initialized without errors
        assert persistence_manager._initialized

        # Database should exist
        assert os.path.exists(persistence_manager.db_path)

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, persistence_manager, sample_session):
        """Test saving and loading a session."""
        # Save session
        await persistence_manager.save_session(sample_session)

        # Load session
        loaded_session = await persistence_manager.load_session(sample_session.id)

        # Verify session data
        assert loaded_session is not None
        assert loaded_session.id == sample_session.id
        assert loaded_session.client_id == sample_session.client_id
        assert loaded_session.prompt == sample_session.prompt
        assert loaded_session.mode == sample_session.mode
        assert loaded_session.current_generation == sample_session.current_generation
        assert loaded_session.status == sample_session.status

        # Verify parameters
        assert loaded_session.parameters.population_size == sample_session.parameters.population_size
        assert loaded_session.parameters.generations == sample_session.parameters.generations

        # Verify fitness weights
        assert loaded_session.fitness_weights.relevance == sample_session.fitness_weights.relevance
        assert loaded_session.fitness_weights.novelty == sample_session.fitness_weights.novelty
        assert loaded_session.fitness_weights.feasibility == sample_session.fitness_weights.feasibility

        # Verify ideas
        assert len(loaded_session.ideas) == len(sample_session.ideas)
        for original, loaded in zip(sample_session.ideas, loaded_session.ideas, strict=False):
            assert loaded.id == original.id
            assert loaded.content == original.content
            assert loaded.generation == original.generation
            assert loaded.fitness == original.fitness

        # Verify workers
        assert len(loaded_session.workers) == len(sample_session.workers)
        for original, loaded in zip(sample_session.workers, loaded_session.workers, strict=False):
            assert loaded.id == original.id
            assert loaded.model == original.model
            assert loaded.status == original.status

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, persistence_manager):
        """Test loading a session that doesn't exist."""
        loaded_session = await persistence_manager.load_session("nonexistent-session")
        assert loaded_session is None

    @pytest.mark.asyncio
    async def test_list_saved_sessions(self, persistence_manager, sample_session):
        """Test listing saved sessions."""
        # Initially no sessions
        sessions = await persistence_manager.list_saved_sessions()
        initial_count = len(sessions)

        # Save a session
        await persistence_manager.save_session(sample_session)

        # List sessions
        sessions = await persistence_manager.list_saved_sessions()
        assert len(sessions) == initial_count + 1

        # Find our session
        our_session = next(s for s in sessions if s["id"] == sample_session.id)
        assert our_session["client_id"] == sample_session.client_id
        assert our_session["mode"] == sample_session.mode.value
        assert our_session["status"] == sample_session.status

    @pytest.mark.asyncio
    async def test_list_sessions_with_client_filter(self, persistence_manager, sample_session):
        """Test listing sessions with client filter."""
        await persistence_manager.save_session(sample_session)

        # List sessions for specific client
        sessions = await persistence_manager.list_saved_sessions(client_id=sample_session.client_id)
        assert len(sessions) >= 1
        assert all(s["client_id"] == sample_session.client_id for s in sessions)

        # List sessions for different client
        sessions = await persistence_manager.list_saved_sessions(client_id="different-client")
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_session(self, persistence_manager, sample_session):
        """Test deleting a session."""
        # Save session
        await persistence_manager.save_session(sample_session)

        # Verify it exists
        loaded = await persistence_manager.load_session(sample_session.id)
        assert loaded is not None

        # Delete session
        success = await persistence_manager.delete_session(sample_session.id)
        assert success

        # Verify it's gone
        loaded = await persistence_manager.load_session(sample_session.id)
        assert loaded is None

        # Try to delete again (should return False)
        success = await persistence_manager.delete_session(sample_session.id)
        assert not success

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, persistence_manager, sample_session):
        """Test saving checkpoints."""
        # Save session first
        await persistence_manager.save_session(sample_session)

        # Save checkpoint
        checkpoint_name = "test_checkpoint"
        additional_data = {"best_fitness": 0.95, "diversity": 0.7}
        await persistence_manager.save_checkpoint(
            sample_session, checkpoint_name, additional_data
        )

        # Load checkpoint
        checkpoint = await persistence_manager.load_checkpoint(
            sample_session.id, checkpoint_name
        )

        assert checkpoint is not None
        assert checkpoint["generation"] == sample_session.current_generation
        assert checkpoint["additional_data"] == additional_data

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, persistence_manager, sample_session):
        """Test listing checkpoints."""
        await persistence_manager.save_session(sample_session)

        # Save multiple checkpoints
        checkpoint_names = ["checkpoint_1", "checkpoint_2", "checkpoint_3"]
        for name in checkpoint_names:
            await persistence_manager.save_checkpoint(sample_session, name)

        # List checkpoints
        checkpoints = await persistence_manager.list_checkpoints(sample_session.id)

        assert len(checkpoints) >= len(checkpoint_names)
        checkpoint_names_found = [cp["checkpoint_name"] for cp in checkpoints]
        for name in checkpoint_names:
            assert name in checkpoint_names_found

    @pytest.mark.asyncio
    async def test_session_with_checkpoint_on_save(self, persistence_manager, sample_session):
        """Test saving a session with a checkpoint name."""
        checkpoint_name = "save_checkpoint"
        await persistence_manager.save_session(sample_session, checkpoint_name)

        # Verify session was saved
        loaded = await persistence_manager.load_session(sample_session.id)
        assert loaded is not None

        # Verify checkpoint was created
        checkpoints = await persistence_manager.list_checkpoints(sample_session.id)
        checkpoint_names = [cp["checkpoint_name"] for cp in checkpoints]
        assert checkpoint_name in checkpoint_names

    @pytest.mark.asyncio
    async def test_database_stats(self, persistence_manager, sample_session):
        """Test getting database statistics."""
        # Get initial stats
        stats = await persistence_manager.get_database_stats()
        initial_sessions = stats["sessions_count"]

        # Save a session
        await persistence_manager.save_session(sample_session)

        # Get updated stats
        stats = await persistence_manager.get_database_stats()
        assert stats["sessions_count"] == initial_sessions + 1
        assert stats["ideas_count"] >= len(sample_session.ideas)
        assert stats["workers_count"] >= len(sample_session.workers)
        assert "database_size_bytes" in stats
        assert stats["database_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_session_with_target_embedding(self, persistence_manager):
        """Test saving and loading session with target embedding."""
        session = Session(
            id="embedding-test",
            client_id="test-client",
            prompt="Test prompt",
            target_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]  # Sample embedding
        )

        # Save and load
        await persistence_manager.save_session(session)
        loaded = await persistence_manager.load_session(session.id)

        assert loaded is not None
        assert loaded.target_embedding == session.target_embedding

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, persistence_manager):
        """Test concurrent save operations."""
        sessions = []
        for i in range(5):
            session = Session(
                id=f"concurrent-session-{i}",
                client_id="concurrent-client",
                prompt=f"Test prompt {i}"
            )
            sessions.append(session)

        # Save all sessions concurrently
        save_tasks = [
            persistence_manager.save_session(session)
            for session in sessions
        ]
        await asyncio.gather(*save_tasks)

        # Verify all sessions were saved
        for session in sessions:
            loaded = await persistence_manager.load_session(session.id)
            assert loaded is not None
            assert loaded.id == session.id
