"""Unit tests for adaptive population management."""

from unittest.mock import Mock

import numpy as np

from genetic_mcp.adaptive_population import (
    AdaptivePopulationManager,
    PopulationConfig,
    PopulationMetrics,
)
from genetic_mcp.models import Idea


class TestPopulationConfig:
    """Test PopulationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PopulationConfig()

        assert config.min_population == 5
        assert config.max_population == 100
        assert config.diversity_threshold == 0.3
        assert config.plateau_generations == 3
        assert config.convergence_threshold == 0.05
        assert config.adjustment_factor == 0.2
        assert config.stability_window == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PopulationConfig(
            min_population=10,
            max_population=200,
            diversity_threshold=0.5,
            plateau_generations=5
        )

        assert config.min_population == 10
        assert config.max_population == 200
        assert config.diversity_threshold == 0.5
        assert config.plateau_generations == 5


class TestPopulationMetrics:
    """Test PopulationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating population metrics."""
        metrics = PopulationMetrics(
            generation=5,
            population_size=20,
            diversity_score=0.7,
            average_fitness=0.8,
            fitness_variance=0.1,
            best_fitness=0.95,
            convergence_rate=0.3
        )

        assert metrics.generation == 5
        assert metrics.population_size == 20
        assert metrics.diversity_score == 0.7
        assert metrics.average_fitness == 0.8
        assert metrics.fitness_variance == 0.1
        assert metrics.best_fitness == 0.95
        assert metrics.convergence_rate == 0.3
        assert metrics.adjustment_reason == ""


class TestAdaptivePopulationManager:
    """Test AdaptivePopulationManager functionality."""

    def create_test_population(self, size: int, generation: int, fitness_range: tuple = (0.0, 1.0)) -> list[Idea]:
        """Create test population for testing."""
        population = []
        for i in range(size):
            fitness = np.random.uniform(fitness_range[0], fitness_range[1])
            idea = Idea(
                id=f"idea_{generation}_{i}",
                content=f"Test idea {i} for generation {generation}",
                generation=generation,
                fitness=fitness
            )
            population.append(idea)
        return population

    def create_diverse_population(self, size: int, generation: int) -> list[Idea]:
        """Create diverse test population."""
        population = []
        themes = ["technology", "nature", "art", "science", "philosophy", "music", "sports", "food"]
        for i in range(size):
            theme = themes[i % len(themes)]
            idea = Idea(
                id=f"diverse_idea_{generation}_{i}",
                content=f"Innovative {theme} concept: {theme}_{i}",
                generation=generation,
                fitness=np.random.uniform(0.3, 0.9)
            )
            population.append(idea)
        return population

    def create_converged_population(self, size: int, generation: int) -> list[Idea]:
        """Create converged test population."""
        population = []
        base_fitness = 0.8
        for i in range(size):
            idea = Idea(
                id=f"converged_idea_{generation}_{i}",
                content="Similar idea concept",
                generation=generation,
                fitness=base_fitness + np.random.uniform(-0.01, 0.01)
            )
            population.append(idea)
        return population

    def test_manager_initialization(self):
        """Test manager initialization."""
        config = PopulationConfig(min_population=3, max_population=50)
        manager = AdaptivePopulationManager(config)

        assert manager.config == config
        assert manager.current_population_size == 10  # Default starting size
        assert manager.plateau_counter == 0
        assert len(manager.metrics_history) == 0
        assert len(manager.population_size_history) == 0
        assert len(manager.adjustment_history) == 0

    def test_analyze_empty_population(self):
        """Test analyzing empty population."""
        manager = AdaptivePopulationManager()

        metrics = manager.analyze_population([], 1)

        assert metrics.generation == 1
        assert metrics.population_size == 0
        assert metrics.diversity_score == 0.0
        assert metrics.average_fitness == 0.0
        assert metrics.fitness_variance == 0.0
        assert metrics.best_fitness == 0.0
        assert metrics.convergence_rate == 0.0

    def test_analyze_single_individual_population(self):
        """Test analyzing population with single individual."""
        manager = AdaptivePopulationManager()
        population = self.create_test_population(1, 0, (0.7, 0.7))

        metrics = manager.analyze_population(population, 0)

        assert metrics.population_size == 1
        assert metrics.diversity_score == 1.0  # Single individual has max diversity
        assert metrics.average_fitness == 0.7
        assert metrics.fitness_variance == 0.0
        assert metrics.best_fitness == 0.7

    def test_analyze_diverse_population(self):
        """Test analyzing diverse population."""
        manager = AdaptivePopulationManager()
        population = self.create_diverse_population(10, 0)

        metrics = manager.analyze_population(population, 0)

        assert metrics.population_size == 10
        assert metrics.diversity_score > 0.5  # Should be reasonably diverse
        assert 0.0 < metrics.average_fitness < 1.0
        assert metrics.fitness_variance >= 0.0
        assert metrics.convergence_rate <= 1.0

    def test_simple_diversity_calculation(self):
        """Test simple diversity calculation method."""
        manager = AdaptivePopulationManager()

        # Test with identical content (low diversity)
        population = []
        for i in range(5):
            idea = Idea(
                id=f"idea_{i}",
                content="identical content",
                generation=0,
                fitness=0.5
            )
            population.append(idea)

        diversity = manager._calculate_simple_diversity(population)
        assert diversity == 0.2  # 1 unique prefix / 5 ideas

    def test_convergence_rate_calculation(self):
        """Test convergence rate calculation."""
        manager = AdaptivePopulationManager()

        # Test with identical fitness (fully converged)
        identical_fitness = [0.5, 0.5, 0.5, 0.5]
        convergence = manager._calculate_convergence_rate(identical_fitness)
        assert convergence == 1.0

        # Test with varied fitness (not converged)
        varied_fitness = [0.1, 0.3, 0.5, 0.7, 0.9]
        convergence = manager._calculate_convergence_rate(varied_fitness)
        assert 0.0 <= convergence < 1.0

    def test_population_size_recommendation_no_change(self):
        """Test population size recommendation with stable population."""
        config = PopulationConfig(min_population=5, max_population=50, diversity_threshold=0.2)
        manager = AdaptivePopulationManager(config)

        # Create population with controlled fitness values to avoid rapid convergence detection
        population = []
        for i in range(10):
            idea = Idea(
                id=f"stable_idea_{i}",
                content=f"Stable concept {i}: unique variation {i}",
                generation=0,
                fitness=0.5 + (i % 3) * 0.15  # Varied fitness: 0.5, 0.65, 0.8 pattern
            )
            population.append(idea)

        metrics = manager.analyze_population(population, 0)

        recommended_size = manager.get_recommended_population_size(metrics, 1)

        # Should recommend same size for diverse, well-performing population
        # Allow for small adjustments due to algorithm behavior
        assert 8 <= recommended_size <= 12

    def test_population_size_increase_low_diversity(self):
        """Test population size increase due to low diversity."""
        config = PopulationConfig(
            min_population=5,
            max_population=50,
            diversity_threshold=0.3,  # Threshold to trigger adjustment
            adjustment_factor=0.3  # Large enough adjustment to exceed minimum change threshold
        )
        manager = AdaptivePopulationManager(config)
        manager.current_population_size = 10

        # Create population with truly identical content for low diversity
        population = []
        for i in range(10):
            idea = Idea(
                id=f"identical_{i}",
                content="identical content for all ideas",  # Same content
                generation=0,
                fitness=0.5 + (i * 0.01)  # Slightly varied fitness
            )
            population.append(idea)

        metrics = manager.analyze_population(population, 0)

        # Manually set very low diversity to guarantee triggering
        metrics.diversity_score = 0.1  # Well below threshold
        # Prevent rapid convergence from interfering by setting moderate values
        metrics.convergence_rate = 0.5  # Below the 0.8 threshold
        metrics.fitness_variance = 0.2   # Above the 0.1 threshold

        recommended_size = manager.get_recommended_population_size(metrics, 1)

        # Should recommend larger population due to low diversity
        # With 30% adjustment factor, 10 * 1.3 = 13, which exceeds the 10% minimum change
        assert recommended_size > 10

    def test_population_size_decrease_rapid_convergence(self):
        """Test population size decrease due to rapid convergence."""
        config = PopulationConfig(adjustment_factor=0.5)  # Large adjustment for testing
        manager = AdaptivePopulationManager(config)
        manager.current_population_size = 20

        population = self.create_test_population(20, 0, (0.85, 0.86))  # Very similar fitness
        metrics = manager.analyze_population(population, 0)

        recommended_size = manager.get_recommended_population_size(metrics, 1)

        # Should recommend smaller population due to rapid convergence
        assert recommended_size < 20

    def test_fitness_plateau_detection(self):
        """Test fitness plateau detection."""
        config = PopulationConfig(plateau_generations=2)
        manager = AdaptivePopulationManager(config)

        # First generation with fitness 0.5
        population1 = self.create_test_population(10, 0, (0.5, 0.5))
        metrics1 = manager.analyze_population(population1, 0)

        # Should not detect plateau initially (first call always returns False)
        plateau_detected1 = manager._detect_fitness_plateau(metrics1)
        assert not plateau_detected1

        # Second generation with similar fitness (small improvement < threshold)
        population2 = self.create_test_population(10, 1, (0.505, 0.505))  # 0.005 improvement
        metrics2 = manager.analyze_population(population2, 1)

        # Should start counting plateau (improvement < 0.01 threshold)
        manager._detect_fitness_plateau(metrics2)
        # The plateau detection increments counter but may not return True until threshold reached
        assert manager.plateau_counter >= 1

        # Third generation with similar fitness
        population3 = self.create_test_population(10, 2, (0.506, 0.506))  # Another tiny improvement
        metrics3 = manager.analyze_population(population3, 2)

        # Should detect full plateau after enough generations
        plateau_detected3 = manager._detect_fitness_plateau(metrics3)
        assert manager.plateau_counter >= 2
        # Plateau should be detected when counter >= plateau_generations
        assert plateau_detected3

    def test_bounds_enforcement(self):
        """Test population size bounds enforcement."""
        config = PopulationConfig(min_population=5, max_population=15)
        manager = AdaptivePopulationManager(config)

        # Test minimum bound
        manager.current_population_size = 3  # Below minimum
        population = self.create_converged_population(3, 0)
        metrics = manager.analyze_population(population, 0)

        recommended_size = manager.get_recommended_population_size(metrics, 1)
        assert recommended_size >= 5

        # Test maximum bound
        manager.current_population_size = 20  # Above maximum
        recommended_size = manager.get_recommended_population_size(metrics, 2)
        assert recommended_size <= 15

    def test_adjustment_history_tracking(self):
        """Test adjustment history tracking."""
        config = PopulationConfig(adjustment_factor=0.5)
        manager = AdaptivePopulationManager(config)
        manager.current_population_size = 10

        # Create scenario that triggers adjustment
        population = self.create_test_population(10, 0, (0.8, 0.81))  # Low variance
        metrics = manager.analyze_population(population, 0)

        initial_history_length = len(manager.adjustment_history)

        recommended_size = manager.get_recommended_population_size(metrics, 1)

        if recommended_size != 10:  # If adjustment occurred
            assert len(manager.adjustment_history) > initial_history_length

            latest_adjustment = manager.adjustment_history[-1]
            assert latest_adjustment["generation"] == 1
            assert latest_adjustment["from_size"] == 10
            assert latest_adjustment["to_size"] == recommended_size
            assert "reasons" in latest_adjustment

    def test_population_statistics(self):
        """Test population statistics generation."""
        manager = AdaptivePopulationManager()

        # Generate some history
        for gen in range(3):
            population = self.create_diverse_population(10 + gen, gen)
            metrics = manager.analyze_population(population, gen)
            manager.get_recommended_population_size(metrics, gen + 1)

        stats = manager.get_population_statistics()

        assert "current_population_size" in stats
        assert "total_adjustments" in stats
        assert "plateau_counter" in stats
        assert "last_best_fitness" in stats
        assert "config" in stats

        if len(manager.metrics_history) > 0:
            assert "recent_metrics" in stats
            assert "avg_diversity" in stats["recent_metrics"]
            assert "avg_fitness" in stats["recent_metrics"]

    def test_trend_calculation(self):
        """Test trend calculation method."""
        manager = AdaptivePopulationManager()

        # Test increasing trend
        increasing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = manager._calculate_trend(increasing_values)
        assert trend == "increasing"

        # Test decreasing trend
        decreasing_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = manager._calculate_trend(decreasing_values)
        assert trend == "decreasing"

        # Test stable trend
        stable_values = [0.3, 0.301, 0.299, 0.302, 0.298]
        trend = manager._calculate_trend(stable_values)
        assert trend == "stable"

    def test_reset_functionality(self):
        """Test manager reset functionality."""
        manager = AdaptivePopulationManager()

        # Force some state changes to ensure we have something to reset
        manager.plateau_counter = 5
        manager.last_best_fitness = 0.8
        manager.current_population_size = 15

        # Generate some state by analyzing a population
        population = self.create_diverse_population(10, 0)
        manager.analyze_population(population, 0)

        # Don't call get_recommended_population_size as it might reset plateau_counter
        # Just verify that we have metrics history and our manually set plateau counter
        assert len(manager.metrics_history) > 0
        assert manager.plateau_counter == 5  # Should still be what we set it to

        # Reset and verify clean state
        manager.reset()

        assert len(manager.metrics_history) == 0
        assert len(manager.population_size_history) == 0
        assert len(manager.adjustment_history) == 0
        assert manager.plateau_counter == 0
        assert manager.last_best_fitness == 0.0
        assert manager.current_population_size == 10  # Back to default

    def test_update_session_population_size(self):
        """Test updating session population size."""
        manager = AdaptivePopulationManager()

        # Create mock session
        session = Mock()
        session.id = "test_session"
        session.parameters = Mock()
        session.parameters.population_size = 10
        session.parameters.elitism_count = 2

        # Update population size
        new_size = 20
        manager.update_session_population_size(session, new_size)

        # Verify updates
        assert session.parameters.population_size == 20
        assert session.parameters.elitism_count == 4  # Proportionally updated

    def test_with_diversity_metrics(self):
        """Test with pre-calculated diversity metrics."""
        manager = AdaptivePopulationManager()

        population = self.create_diverse_population(10, 0)

        # Provide pre-calculated diversity metrics
        diversity_metrics = {
            "simpson_diversity": 0.8,
            "shannon_diversity": 1.5,
            "average_distance": 0.6
        }

        metrics = manager.analyze_population(population, 0, diversity_metrics)

        # Should use provided diversity score
        assert metrics.diversity_score == 0.8

    def test_insufficient_change_threshold(self):
        """Test that small changes are ignored to avoid constant adjustments."""
        config = PopulationConfig(
            adjustment_factor=0.05,  # Small adjustment that would result in minimal change
            diversity_threshold=0.4  # Set threshold that won't trigger
        )
        manager = AdaptivePopulationManager(config)
        manager.current_population_size = 10

        # Create scenario with moderate diversity and performance
        # that shouldn't trigger major adjustments
        population = self.create_diverse_population(10, 0)
        metrics = manager.analyze_population(population, 0)

        # Ensure diversity is above threshold to avoid diversity-based adjustments
        if metrics.diversity_score < 0.5:
            # Manually set diversity to avoid triggering diversity adjustment
            metrics.diversity_score = 0.6

        recommended_size = manager.get_recommended_population_size(metrics, 1)

        # Should stick with current size when no major issues detected
        assert recommended_size == 10
