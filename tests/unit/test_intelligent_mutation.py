"""Tests for intelligent mutation strategies."""

from unittest.mock import AsyncMock

import pytest

from genetic_mcp.intelligent_mutation import (
    ComponentMutationRate,
    FitnessLandscape,
    IntelligentMutationManager,
    MutationMetrics,
    MutationStrategy,
)
from genetic_mcp.models import Idea


class TestIntelligentMutationManager:
    """Test the IntelligentMutationManager class."""

    @pytest.fixture
    def mutation_manager(self):
        """Create a mutation manager for testing."""
        llm_client = AsyncMock()
        llm_client.generate = AsyncMock(return_value="Mutated idea content")
        return IntelligentMutationManager(llm_client=llm_client)

    @pytest.fixture
    def sample_idea(self):
        """Create a sample idea for testing."""
        return Idea(
            id="test_idea_1",
            content="A system for automated task management using AI algorithms",
            generation=1,
            fitness=0.7
        )

    @pytest.fixture
    def sample_population(self):
        """Create a sample population for testing."""
        return [
            Idea(
                id=f"idea_{i}",
                content=f"Idea number {i} with different content and approach",
                generation=1,
                fitness=0.5 + (i * 0.1)
            )
            for i in range(5)
        ]

    def test_mutation_manager_initialization(self, mutation_manager):
        """Test mutation manager initializes correctly."""
        assert mutation_manager.llm_client is not None
        assert len(mutation_manager.mutation_history) == 0
        assert len(mutation_manager.successful_patterns) == 0
        assert len(mutation_manager.component_rates) == 0
        assert mutation_manager.temperature == 1.0

    def test_strategy_selection(self, mutation_manager, sample_idea):
        """Test strategy selection logic."""
        # Early generation should prefer exploration
        strategy = mutation_manager._select_optimal_strategy(sample_idea, generation=1)
        assert strategy in [MutationStrategy.RANDOM, MutationStrategy.CONTEXT_AWARE, MutationStrategy.COMPONENT_BASED]

    @pytest.mark.asyncio
    async def test_basic_mutation(self, mutation_manager, sample_idea, sample_population):
        """Test basic mutation functionality."""
        result = await mutation_manager.mutate(
            idea=sample_idea,
            all_ideas=sample_population,
            generation=1,
            strategy=MutationStrategy.RANDOM
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert len(mutation_manager.mutation_history) == 1

    @pytest.mark.asyncio
    async def test_guided_mutation(self, mutation_manager, sample_idea, sample_population):
        """Test guided mutation with landscape analysis."""
        result = await mutation_manager.mutate(
            idea=sample_idea,
            all_ideas=sample_population,
            generation=3,
            strategy=MutationStrategy.GUIDED
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_component_extraction(self, mutation_manager):
        """Test component extraction from idea content."""
        content = "The problem is complex. Our solution uses advanced algorithms. Implementation will be challenging."
        components = mutation_manager._extract_components(content)

        assert isinstance(components, dict)
        assert len(components) > 0

    def test_fitness_feedback_update(self, mutation_manager, sample_idea):
        """Test fitness feedback updates mutation metrics."""
        # Create a mock mutation history entry
        metric = MutationMetrics(
            strategy="random",
            fitness_before=0.5,
            fitness_after=0.5,  # Will be updated
            generation=1,
            component_modified="test_component"
        )
        metric.idea_id = sample_idea.id
        mutation_manager.mutation_history.append(metric)

        # Update fitness feedback
        mutation_manager.update_mutation_feedback(sample_idea.id, 0.8)

        # Check that metrics were updated
        updated_metric = mutation_manager.mutation_history[-1]
        assert updated_metric.fitness_after == 0.8
        assert abs(updated_metric.improvement - 0.3) < 0.001  # Allow for floating point precision
        assert updated_metric.success

    def test_component_mutation_rate_adaptation(self):
        """Test component mutation rate adaptation."""
        rate = ComponentMutationRate("test_component")
        initial_rate = rate.current_rate

        # Success should increase rate
        rate.update_rate(success=True)
        assert rate.current_rate > initial_rate

        # Failure should decrease rate
        rate.update_rate(success=False)
        assert rate.current_rate < initial_rate

    def test_fitness_landscape_analysis(self):
        """Test fitness landscape analysis."""
        center_idea = Idea(id="center", content="Center idea", fitness=0.7)
        neighbors = [
            Idea(id="n1", content="Neighbor 1", fitness=0.8),
            Idea(id="n2", content="Neighbor 2", fitness=0.6),
            Idea(id="n3", content="Neighbor 3", fitness=0.9)
        ]

        landscape = FitnessLandscape(center_idea=center_idea)
        landscape.analyze_landscape(neighbors)

        assert len(landscape.neighboring_points) == 3
        assert len(landscape.promising_directions) > 0  # Should have high-fitness neighbors

    def test_performance_report(self, mutation_manager):
        """Test performance report generation."""
        # Add some mock history
        metric = MutationMetrics(
            strategy="random",
            fitness_before=0.5,
            fitness_after=0.7,
            generation=1,
            component_modified="test_component"
        )
        mutation_manager.mutation_history.append(metric)

        report = mutation_manager.get_performance_report()

        assert isinstance(report, dict)
        assert 'strategy_performance' in report
        assert 'component_rates' in report
        assert 'total_mutations' in report
        assert report['total_mutations'] == 1

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, sample_idea, sample_population):
        """Test fallback behavior when LLM calls fail."""
        # Create manager with failing LLM client
        llm_client = AsyncMock()
        llm_client.generate = AsyncMock(side_effect=Exception("LLM failed"))
        mutation_manager = IntelligentMutationManager(llm_client=llm_client)

        result = await mutation_manager.mutate(
            idea=sample_idea,
            all_ideas=sample_population,
            generation=1,
            strategy=MutationStrategy.GUIDED
        )

        # Should still return a valid result (fallback)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_population_theme_analysis(self, mutation_manager, sample_population):
        """Test population theme analysis."""
        themes = mutation_manager._analyze_population_themes(sample_population)

        assert isinstance(themes, dict)
        assert len(themes) > 0
        # Should contain words from the sample ideas
        assert any('idea' in theme.lower() for theme in themes)

    def test_successful_pattern_learning(self, mutation_manager):
        """Test learning from successful mutation patterns."""
        # Create a successful mutation
        metric = MutationMetrics(
            strategy="guided",
            fitness_before=0.5,
            fitness_after=0.8,  # Significant improvement
            generation=1,
            component_modified="solution_approach"
        )

        mutation_manager._learn_successful_pattern(metric)

        assert "guided" in mutation_manager.successful_patterns
        assert len(mutation_manager.successful_patterns["guided"]) > 0

    def test_reset_adaptation(self, mutation_manager):
        """Test resetting adaptation state."""
        # Add some history and state
        mutation_manager.temperature = 0.5
        mutation_manager.mutation_history.append(
            MutationMetrics("test", 0.5, 0.7, 1, "test_component")
        )

        mutation_manager.reset_adaptation()

        assert mutation_manager.temperature == 1.0
        assert len(mutation_manager.mutation_history) == 0
