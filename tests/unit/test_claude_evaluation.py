"""Unit tests for Claude evaluation functionality."""

import numpy as np
import pytest

from genetic_mcp.fitness import FitnessEvaluator
from genetic_mcp.models import Idea, Session


class TestClaudeEvaluation:
    """Test Claude evaluation integration with fitness calculation."""

    def test_idea_claude_fields(self):
        """Test that Idea model includes Claude evaluation fields."""
        idea = Idea(
            id="test-1",
            content="Test idea",
            generation=0
        )

        # Check default values
        assert idea.claude_evaluation is None
        assert idea.claude_score is None
        assert idea.combined_fitness is None

        # Set Claude evaluation
        idea.claude_evaluation = {
            "score": 0.85,
            "justification": "Excellent idea",
            "strengths": ["Innovative", "Practical"]
        }
        idea.claude_score = 0.85
        idea.combined_fitness = 0.8

        assert idea.claude_score == 0.85
        assert idea.combined_fitness == 0.8
        assert idea.claude_evaluation["justification"] == "Excellent idea"

    def test_session_claude_flags(self):
        """Test that Session model includes Claude evaluation flags."""
        session = Session(
            id="test-session",
            client_id="test-client",
            prompt="Test prompt"
        )

        # Check defaults
        assert session.claude_evaluation_enabled is False
        assert session.claude_evaluation_weight == 0.5

        # Enable Claude evaluation
        session.claude_evaluation_enabled = True
        session.claude_evaluation_weight = 0.7

        assert session.claude_evaluation_enabled is True
        assert session.claude_evaluation_weight == 0.7

    def test_fitness_with_claude_score(self):
        """Test fitness calculation with Claude evaluation."""
        evaluator = FitnessEvaluator()

        # Create test ideas
        idea1 = Idea(id="1", content="Idea 1", generation=0)
        idea2 = Idea(id="2", content="Idea 2", generation=0)
        ideas = [idea1, idea2]

        # Add embeddings
        target_embedding = np.random.rand(100).tolist()
        evaluator.add_embedding("1", np.random.rand(100).tolist())
        evaluator.add_embedding("2", np.random.rand(100).tolist())

        # Calculate fitness without Claude
        fitness1 = evaluator.calculate_fitness(idea1, ideas, target_embedding)
        assert idea1.fitness == fitness1
        assert idea1.combined_fitness is None

        # Add Claude score and recalculate with weight
        idea1.claude_score = 0.9
        claude_weight = 0.4

        combined_fitness = evaluator.calculate_fitness(
            idea1, ideas, target_embedding, claude_weight
        )

        # Check combined fitness calculation
        expected_combined = (1 - claude_weight) * idea1.fitness + claude_weight * 0.9
        assert idea1.combined_fitness == pytest.approx(expected_combined, 0.001)
        assert combined_fitness == idea1.combined_fitness

    def test_evaluate_population_with_claude(self):
        """Test population evaluation with Claude scores."""
        evaluator = FitnessEvaluator()

        # Create population
        ideas = [
            Idea(id=f"idea-{i}", content=f"Content {i}", generation=0)
            for i in range(5)
        ]

        # Add embeddings and Claude scores for some ideas
        target_embedding = np.random.rand(100).tolist()
        for i, idea in enumerate(ideas):
            evaluator.add_embedding(idea.id, np.random.rand(100).tolist())
            if i < 3:  # Only first 3 have Claude scores
                idea.claude_score = 0.7 + i * 0.1

        # Evaluate with Claude weight
        evaluator.evaluate_population(ideas, target_embedding, claude_evaluation_weight=0.5)

        # Check that ideas with Claude scores have combined fitness
        for i, idea in enumerate(ideas):
            if i < 3:
                assert idea.combined_fitness is not None
                # Combined should be between algorithmic and Claude scores
                assert min(idea.fitness, idea.claude_score) <= idea.combined_fitness <= max(idea.fitness, idea.claude_score)
            else:
                assert idea.combined_fitness is None

    def test_fitness_sorting_with_combined_scores(self):
        """Test that ideas are properly sorted by combined fitness when available."""
        ideas = []
        for i in range(5):
            idea = Idea(id=f"idea-{i}", content=f"Content {i}", generation=0)
            idea.fitness = 0.5 + i * 0.05  # 0.5, 0.55, 0.6, 0.65, 0.7

            # Give middle ideas high Claude scores
            if i in [1, 2]:
                idea.claude_score = 0.9
                idea.combined_fitness = 0.8  # Higher than their algorithmic fitness

            ideas.append(idea)

        # Sort by effective fitness (combined if available, else regular)
        sorted_ideas = sorted(
            ideas,
            key=lambda x: x.combined_fitness if x.combined_fitness is not None else x.fitness,
            reverse=True
        )

        # Ideas 1 and 2 should be at the top due to high combined fitness
        assert sorted_ideas[0].id in ["idea-1", "idea-2"]
        assert sorted_ideas[1].id in ["idea-1", "idea-2"]

        # Idea 4 (highest algorithmic) should be third
        assert sorted_ideas[2].id == "idea-4"
