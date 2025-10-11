"""Unit tests for data models."""

from datetime import datetime

import pytest

from genetic_mcp.models import (
    EvolutionMode,
    FitnessWeights,
    GenerationRequest,
    GeneticParameters,
    Idea,
    Session,
    Worker,
    WorkerStatus,
)


class TestIdea:
    """Test the Idea model."""

    def test_idea_creation(self):
        """Test creating an idea."""
        idea = Idea(
            id="test-1",
            content="Test idea content",
            generation=1,
            parent_ids=["parent-1", "parent-2"]
        )

        assert idea.id == "test-1"
        assert idea.content == "Test idea content"
        assert idea.generation == 1
        assert idea.parent_ids == ["parent-1", "parent-2"]
        assert idea.fitness == 0.0
        assert isinstance(idea.created_at, datetime)

    def test_idea_with_scores(self):
        """Test idea with fitness scores."""
        idea = Idea(
            id="test-2",
            content="Another idea",
            scores={"relevance": 0.8, "novelty": 0.6, "feasibility": 0.7},
            fitness=0.7
        )

        assert idea.scores["relevance"] == 0.8
        assert idea.scores["novelty"] == 0.6
        assert idea.scores["feasibility"] == 0.7
        assert idea.fitness == 0.7


class TestWorker:
    """Test the Worker model."""

    def test_worker_creation(self):
        """Test creating a worker."""
        worker = Worker(
            id="worker-1",
            model="openai"
        )

        assert worker.id == "worker-1"
        assert worker.model == "openai"
        assert worker.status == WorkerStatus.IDLE
        assert worker.completed_tasks == 0
        assert worker.failed_tasks == 0

    def test_worker_status_update(self):
        """Test updating worker status."""
        worker = Worker(id="worker-2", model="anthropic")

        worker.status = WorkerStatus.WORKING
        worker.current_task = "task-1"

        assert worker.status == WorkerStatus.WORKING
        assert worker.current_task == "task-1"


class TestFitnessWeights:
    """Test the FitnessWeights model."""

    def test_default_weights(self):
        """Test default fitness weights."""
        weights = FitnessWeights()

        assert weights.relevance == 0.4
        assert weights.novelty == 0.3
        assert weights.feasibility == 0.3
        assert abs(weights.relevance + weights.novelty + weights.feasibility - 1.0) < 0.001

    def test_custom_weights(self):
        """Test custom fitness weights."""
        weights = FitnessWeights(
            relevance=0.5,
            novelty=0.3,
            feasibility=0.2
        )

        assert weights.relevance == 0.5
        assert weights.novelty == 0.3
        assert weights.feasibility == 0.2

    def test_invalid_weights_sum(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Base weights .* must sum to 1.0"):
            FitnessWeights(
                relevance=0.5,
                novelty=0.5,
                feasibility=0.5
            )

    def test_invalid_weight_range(self):
        """Test that weights must be between 0 and 1."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            FitnessWeights(
                relevance=1.5,
                novelty=0.3,
                feasibility=-0.8
            )


class TestGeneticParameters:
    """Test the GeneticParameters model."""

    def test_default_parameters(self):
        """Test default genetic parameters."""
        params = GeneticParameters()

        assert params.population_size == 10
        assert params.generations == 5
        assert params.mutation_rate == 0.1
        assert params.crossover_rate == 0.7
        assert params.elitism_count == 2

    def test_custom_parameters(self):
        """Test custom genetic parameters."""
        params = GeneticParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_count=4
        )

        assert params.population_size == 20
        assert params.generations == 10
        assert params.mutation_rate == 0.2
        assert params.crossover_rate == 0.8
        assert params.elitism_count == 4

    def test_invalid_elitism_count(self):
        """Test that elitism count must be at most half of population."""
        with pytest.raises(ValueError, match="Elitism count .* must be at most half"):
            GeneticParameters(
                population_size=10,
                elitism_count=6
            )


class TestSession:
    """Test the Session model."""

    def test_session_creation(self):
        """Test creating a session."""
        session = Session(
            id="session-1",
            client_id="client-1",
            prompt="Test prompt"
        )

        assert session.id == "session-1"
        assert session.client_id == "client-1"
        assert session.prompt == "Test prompt"
        assert session.mode == EvolutionMode.SINGLE_PASS
        assert session.status == "active"
        assert session.current_generation == 0

    def test_get_top_ideas(self):
        """Test getting top ideas by fitness."""
        session = Session(
            id="session-2",
            client_id="client-2",
            prompt="Test"
        )

        # Add ideas with different fitness scores
        ideas = [
            Idea(id="1", content="Idea 1", fitness=0.5),
            Idea(id="2", content="Idea 2", fitness=0.8),
            Idea(id="3", content="Idea 3", fitness=0.3),
            Idea(id="4", content="Idea 4", fitness=0.9),
        ]
        session.ideas = ideas

        top_2 = session.get_top_ideas(2)
        assert len(top_2) == 2
        assert top_2[0].fitness == 0.9
        assert top_2[1].fitness == 0.8

    def test_get_active_workers(self):
        """Test getting active workers."""
        session = Session(
            id="session-3",
            client_id="client-3",
            prompt="Test"
        )

        workers = [
            Worker(id="w1", model="m1", status=WorkerStatus.IDLE),
            Worker(id="w2", model="m2", status=WorkerStatus.WORKING),
            Worker(id="w3", model="m3", status=WorkerStatus.WORKING),
            Worker(id="w4", model="m4", status=WorkerStatus.COMPLETED),
        ]
        session.workers = workers

        active = session.get_active_workers()
        assert len(active) == 2
        assert all(w.status == WorkerStatus.WORKING for w in active)


class TestGenerationRequest:
    """Test the GenerationRequest model."""

    def test_default_request(self):
        """Test default generation request."""
        request = GenerationRequest(
            prompt="Test prompt"
        )

        assert request.prompt == "Test prompt"
        assert request.mode == EvolutionMode.SINGLE_PASS
        assert request.population_size == 10
        assert request.top_k == 5
        assert request.generations == 5

    def test_custom_request(self):
        """Test custom generation request."""
        request = GenerationRequest(
            prompt="Test prompt",
            mode=EvolutionMode.ITERATIVE,
            population_size=20,
            top_k=10,
            generations=8,
            fitness_weights={"relevance": 0.6, "novelty": 0.2, "feasibility": 0.2}
        )

        assert request.mode == EvolutionMode.ITERATIVE
        assert request.population_size == 20
        assert request.top_k == 10
        assert request.generations == 8
        assert request.fitness_weights is not None
