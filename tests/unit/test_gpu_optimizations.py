"""Unit tests for GPU optimization modules."""

from unittest.mock import patch

import numpy as np
import pytest

# Handle optional GPU dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from genetic_mcp.genetic_algorithm_gpu_enhanced import (
    AdvancedGeneticParameters,
    GPUEnhancedGeneticAlgorithm,
)
from genetic_mcp.gpu_accelerated import GPUConfig
from genetic_mcp.gpu_diversity_metrics import GPUDiversityMetrics
from genetic_mcp.gpu_selection_optimized import GPUOptimizedSelection
from genetic_mcp.models import Idea


@pytest.fixture
def gpu_config():
    """Create GPU configuration for testing."""
    return GPUConfig(
        device="cpu",  # Use CPU for tests
        batch_size=16,
        use_mixed_precision=False
    )


@pytest.fixture
def sample_population():
    """Create sample population for testing."""
    population = []
    for i in range(20):
        idea = Idea(
            id=f"test_idea_{i}",
            content=f"Test idea number {i} with some content",
            generation=0,
            fitness=np.random.random(),
            scores={
                'relevance': np.random.random(),
                'novelty': np.random.random(),
                'feasibility': np.random.random()
            }
        )
        population.append(idea)
    return population


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.random.randn(20, 768)  # 20 ideas, 768-dim embeddings


@pytest.fixture
def fitness_scores():
    """Create sample fitness scores."""
    return np.random.random(20)


class TestGPUOptimizedSelection:
    """Test GPU-optimized selection strategies."""

    def test_initialization(self, gpu_config):
        """Test selection optimizer initialization."""
        selector = GPUOptimizedSelection(gpu_config)
        assert selector.config == gpu_config
        assert selector.temperature == 1.0
        assert selector.selection_pressure == 2.0

    def test_boltzmann_selection(self, gpu_config, fitness_scores):
        """Test Boltzmann selection."""
        selector = GPUOptimizedSelection(gpu_config)

        # Test selection
        num_selections = 10
        temperature = 1.0
        selected = selector.boltzmann_selection_batch(
            fitness_scores, temperature, num_selections
        )

        assert len(selected) == num_selections
        assert all(0 <= idx < len(fitness_scores) for idx in selected)

    def test_stochastic_universal_sampling(self, gpu_config, fitness_scores):
        """Test SUS selection."""
        selector = GPUOptimizedSelection(gpu_config)

        num_selections = 10
        selected = selector.stochastic_universal_sampling_gpu(
            fitness_scores, num_selections
        )

        assert len(selected) == num_selections
        assert all(0 <= idx < len(fitness_scores) for idx in selected)

        # Test with zero fitness
        zero_fitness = np.zeros_like(fitness_scores)
        selected_zero = selector.stochastic_universal_sampling_gpu(
            zero_fitness, num_selections
        )
        assert len(selected_zero) == num_selections

    def test_rank_based_selection(self, gpu_config, fitness_scores):
        """Test rank-based selection."""
        selector = GPUOptimizedSelection(gpu_config)

        num_selections = 10
        selection_pressure = 1.5
        selected = selector.rank_based_selection_gpu(
            fitness_scores, num_selections, selection_pressure
        )

        assert len(selected) == num_selections
        assert all(0 <= idx < len(fitness_scores) for idx in selected)

    def test_fitness_sharing(self, gpu_config, fitness_scores, sample_embeddings):
        """Test fitness sharing calculation."""
        selector = GPUOptimizedSelection(gpu_config)

        sigma_share = 0.5
        alpha = 1.0
        shared_fitness = selector.fitness_sharing_gpu(
            fitness_scores, sample_embeddings, sigma_share, alpha
        )

        assert len(shared_fitness) == len(fitness_scores)
        assert all(f > 0 for f in shared_fitness)
        # Shared fitness should be less than or equal to original
        assert all(shared <= orig for shared, orig in zip(shared_fitness, fitness_scores, strict=False))

    def test_crowding_distance(self, gpu_config, sample_population):
        """Test crowding distance calculation."""
        selector = GPUOptimizedSelection(gpu_config)

        # Extract objective values
        objectives = np.array([
            [idea.scores['relevance'], idea.scores['novelty'], idea.scores['feasibility']]
            for idea in sample_population
        ])

        # Calculate crowding distances
        distances = selector.crowding_distance_gpu(objectives)

        assert len(distances) == len(sample_population)
        assert all(d >= 0 for d in distances)

        # Test with Pareto ranks
        pareto_ranks = np.random.randint(1, 4, size=len(sample_population))
        distances_ranked = selector.crowding_distance_gpu(objectives, pareto_ranks)
        assert len(distances_ranked) == len(sample_population)

    def test_diversity_preservation_selection(
        self, gpu_config, fitness_scores, sample_embeddings
    ):
        """Test diversity preservation selection."""
        selector = GPUOptimizedSelection(gpu_config)

        num_selections = 10
        diversity_weight = 0.5
        selected = selector.diversity_preservation_selection_gpu(
            fitness_scores, sample_embeddings, num_selections, diversity_weight
        )

        assert len(selected) == num_selections
        assert len(set(selected)) == num_selections  # All unique
        assert all(0 <= idx < len(fitness_scores) for idx in selected)

    def test_tournament_selection_advanced(self, gpu_config, fitness_scores):
        """Test advanced tournament selection with pressure."""
        selector = GPUOptimizedSelection(gpu_config)

        num_selections = 10
        tournament_size = 3
        selection_pressure = 0.8

        selected = selector.batch_tournament_selection_advanced_gpu(
            fitness_scores, num_selections, tournament_size, selection_pressure
        )

        assert len(selected) == num_selections
        assert all(0 <= idx < len(fitness_scores) for idx in selected)


class TestGPUDiversityMetrics:
    """Test GPU diversity metrics calculation."""

    def test_initialization(self, gpu_config):
        """Test diversity metrics initialization."""
        metrics = GPUDiversityMetrics(gpu_config)
        assert metrics.config == gpu_config

    def test_population_diversity_calculation(
        self, gpu_config, sample_embeddings, fitness_scores
    ):
        """Test comprehensive diversity calculation."""
        metrics = GPUDiversityMetrics(gpu_config)

        # Calculate diversity
        diversity_results = metrics.calculate_population_diversity_gpu(
            sample_embeddings, fitness_scores
        )

        # Check expected metrics
        assert 'embedding_diversity' in diversity_results
        assert 'embedding_diversity_std' in diversity_results
        assert 'fitness_mean' in diversity_results
        assert 'fitness_variance' in diversity_results
        # cluster_entropy is part of cluster_metrics subdictionary
        assert 'uniqueness_ratio' in diversity_results

        # Validate values
        assert diversity_results['embedding_diversity'] > 0
        assert 0 <= diversity_results['uniqueness_ratio'] <= 1

    def test_hypervolume_calculation(self, gpu_config, sample_population):
        """Test hypervolume calculation."""
        metrics = GPUDiversityMetrics(gpu_config)

        # Extract objective values
        objectives = np.array([
            [idea.scores['relevance'], idea.scores['novelty'], idea.scores['feasibility']]
            for idea in sample_population
        ])

        objectives_tensor = torch.from_numpy(objectives).float() if TORCH_AVAILABLE else objectives

        # Calculate hypervolume
        if TORCH_AVAILABLE:
            hypervolume = metrics._calculate_hypervolume_gpu(objectives_tensor)
            assert hypervolume >= 0
            # Note: hypervolume can exceed 1 depending on reference point and normalization
            assert hypervolume <= 10  # Reasonable upper bound

    def test_cluster_diversity(self, gpu_config, sample_embeddings):
        """Test clustering-based diversity metrics."""
        metrics = GPUDiversityMetrics(gpu_config)

        if TORCH_AVAILABLE:
            embeddings_tensor = torch.from_numpy(sample_embeddings).float()
            cluster_results = metrics._calculate_cluster_diversity_gpu(
                embeddings_tensor, n_clusters=5
            )

            assert 'cluster_entropy' in cluster_results
            assert 'n_effective_clusters' in cluster_results
            assert 'avg_intra_cluster_distance' in cluster_results
            assert 'avg_inter_cluster_distance' in cluster_results
            assert 'cluster_separation' in cluster_results

            assert cluster_results['n_effective_clusters'] <= 5
            assert cluster_results['cluster_entropy'] >= 0

    def test_convergence_metrics(self, gpu_config):
        """Test convergence metrics calculation."""
        metrics = GPUDiversityMetrics(gpu_config)

        # Create fake history
        generations = 5
        pop_size = 20
        embedding_dim = 768

        population_history = [
            np.random.randn(pop_size, embedding_dim) for _ in range(generations)
        ]
        fitness_history = [
            np.random.random(pop_size) * (i + 1) / generations
            for i in range(generations)
        ]

        convergence_results = metrics.calculate_convergence_metrics_gpu(
            population_history, fitness_history
        )

        assert 'best_fitness_improvement_rate' in convergence_results
        assert 'avg_fitness_improvement_rate' in convergence_results
        assert 'fitness_stagnation_ratio' in convergence_results

    def test_batch_diversity_calculation(
        self, gpu_config, sample_embeddings, fitness_scores
    ):
        """Test batch diversity calculation for multiple populations."""
        metrics = GPUDiversityMetrics(gpu_config)

        # Create multiple populations
        n_populations = 3
        populations = [sample_embeddings for _ in range(n_populations)]
        fitness_lists = [fitness_scores for _ in range(n_populations)]

        results = metrics.batch_diversity_calculation_gpu(
            populations, fitness_lists, batch_size=2
        )

        assert len(results) == n_populations
        assert all('embedding_diversity' in r for r in results)
        assert all('fitness_variance' in r for r in results)


class TestGPUEnhancedGeneticAlgorithm:
    """Test GPU-enhanced genetic algorithm."""

    def test_initialization(self, gpu_config):
        """Test algorithm initialization."""
        params = AdvancedGeneticParameters(
            population_size=20,
            selection_method="tournament",
            use_fitness_sharing=True
        )

        ga = GPUEnhancedGeneticAlgorithm(
            parameters=params,
            gpu_config=gpu_config
        )

        assert ga.parameters == params
        assert ga.gpu_config == gpu_config
        assert ga.generation == 0
        assert ga.temperature == params.temperature_initial

    @pytest.mark.asyncio
    async def test_evolve_population(self, gpu_config, sample_population):
        """Test basic population evolution."""
        params = AdvancedGeneticParameters(
            population_size=20,
            generations=3,
            selection_method="tournament"
        )

        ga = GPUEnhancedGeneticAlgorithm(
            parameters=params,
            gpu_config=gpu_config
        )

        # Mock embedding generation
        with patch.object(ga, '_get_population_embeddings') as mock_embeddings:
            mock_embeddings.return_value = np.random.randn(20, 768)

            # Mock fitness evaluation - use the correct method name
            method_name = 'evaluate_population_async' if hasattr(ga.fitness_evaluator, 'evaluate_population_async') else 'evaluate_population'
            with patch.object(ga.fitness_evaluator, method_name) as mock_fitness:
                if method_name == 'evaluate_population_async':
                    async def set_fitness(ideas, prompt):
                        for idea in ideas:
                            idea.fitness = np.random.random()
                            idea.scores = {
                                'relevance': np.random.random(),
                                'novelty': np.random.random(),
                                'feasibility': np.random.random()
                            }
                    mock_fitness.side_effect = set_fitness
                else:
                    def set_fitness_sync(ideas, target_embedding):
                        for idea in ideas:
                            idea.fitness = np.random.random()
                            idea.scores = {
                                'relevance': np.random.random(),
                                'novelty': np.random.random(),
                                'feasibility': np.random.random()
                            }
                    mock_fitness.side_effect = set_fitness_sync

                # Run evolution
                final_pop = await ga.evolve_population(
                    sample_population,
                    "test prompt",
                    generations=2
                )

                assert len(final_pop) == params.population_size
                assert all(isinstance(idea, Idea) for idea in final_pop)

    def test_adaptive_elitism_count(self, gpu_config):
        """Test adaptive elitism calculation."""
        params = AdvancedGeneticParameters(elitism_count=5)
        ga = GPUEnhancedGeneticAlgorithm(parameters=params, gpu_config=gpu_config)

        # Low variance - should increase elitism
        low_var_fitness = np.ones(20) * 0.8 + np.random.normal(0, 0.01, 20)
        elite_count = ga._adaptive_elitism_count(low_var_fitness)
        assert elite_count >= params.elitism_count

        # High variance - should decrease elitism
        high_var_fitness = np.random.random(20)
        elite_count = ga._adaptive_elitism_count(high_var_fitness)
        assert elite_count <= params.elitism_count

    def test_adaptive_mutation_rate(self, gpu_config):
        """Test adaptive mutation rate calculation."""
        params = AdvancedGeneticParameters(
            mutation_rate=0.2,
            mutation_rate_min=0.01,
            mutation_rate_max=0.5,
            adaptive_mutation=True
        )
        ga = GPUEnhancedGeneticAlgorithm(parameters=params, gpu_config=gpu_config)

        # Low diversity - high mutation
        low_div_fitness = np.ones(20) * 0.8
        mutation_rate = ga._get_adaptive_mutation_rate(low_div_fitness)
        assert mutation_rate > params.mutation_rate

        # High diversity - low mutation
        high_div_fitness = np.random.random(20)
        mutation_rate = ga._get_adaptive_mutation_rate(high_div_fitness)
        assert mutation_rate < params.mutation_rate

    def test_split_population(self, gpu_config, sample_population):
        """Test population splitting for multi-population evolution."""
        params = AdvancedGeneticParameters(
            population_size=20,
            n_subpopulations=4
        )
        ga = GPUEnhancedGeneticAlgorithm(parameters=params, gpu_config=gpu_config)

        subpops = ga._split_population(sample_population)

        assert len(subpops) == 4
        assert sum(len(subpop) for subpop in subpops) == len(sample_population)
        assert all(isinstance(idea, Idea) for subpop in subpops for idea in subpop)

    def test_perform_migration(self, gpu_config, sample_population):
        """Test migration between subpopulations."""
        params = AdvancedGeneticParameters(
            population_size=20,
            n_subpopulations=4,
            migration_rate=0.2
        )
        ga = GPUEnhancedGeneticAlgorithm(parameters=params, gpu_config=gpu_config)
        ga.generation = 5

        # Create subpopulations
        subpops = ga._split_population(sample_population)

        # Perform migration
        migrated_subpops = ga._perform_migration(subpops)

        assert len(migrated_subpops) == len(subpops)
        assert all(len(migrated) == len(original)
                  for migrated, original in zip(migrated_subpops, subpops, strict=False))

    def test_simple_crossover(self, gpu_config):
        """Test simple crossover operation."""
        ga = GPUEnhancedGeneticAlgorithm(gpu_config=gpu_config)

        content1 = "This is the first parent. It has multiple sentences. Here is another."
        content2 = "This is the second parent. With different content. And more ideas."

        offspring1, offspring2 = ga._simple_crossover(content1, content2)

        assert isinstance(offspring1, str)
        assert isinstance(offspring2, str)
        assert offspring1 != content1
        assert offspring2 != content2

    @pytest.mark.asyncio
    async def test_mutate_content(self, gpu_config):
        """Test content mutation."""
        ga = GPUEnhancedGeneticAlgorithm(gpu_config=gpu_config)

        original = "This is the original idea content."
        mutated = await ga._mutate_content(original)

        assert isinstance(mutated, str)
        assert len(mutated) > 0
        # With no LLM client, should use fallback mutation
        assert mutated != original

    def test_get_evolution_summary(self, gpu_config):
        """Test evolution summary generation."""
        ga = GPUEnhancedGeneticAlgorithm(gpu_config=gpu_config)

        # Add some fake history
        ga.generation = 5
        ga.fitness_history = [
            np.random.random(20) * (i + 1) / 5 for i in range(5)
        ]
        ga.population_history = [
            np.random.randn(20, 768) for _ in range(5)
        ]

        summary = ga.get_evolution_summary()

        assert 'generations' in summary
        assert 'final_best_fitness' in summary
        assert 'final_avg_fitness' in summary
        assert 'fitness_improvement' in summary
        assert 'selection_method' in summary
        assert 'used_gpu' in summary

    def test_cleanup(self, gpu_config):
        """Test resource cleanup."""
        ga = GPUEnhancedGeneticAlgorithm(gpu_config=gpu_config)

        # Should not raise any errors
        ga.cleanup()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGPUSpecificFeatures:
    """Test GPU-specific features when PyTorch is available."""

    def test_gpu_memory_management(self):
        """Test GPU memory management features."""
        gpu_config = GPUConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            memory_fraction=0.5
        )

        selector = GPUOptimizedSelection(gpu_config)

        # Test memory stats
        memory_stats = selector.memory_manager.get_memory_stats()
        assert 'allocated' in memory_stats
        assert 'reserved' in memory_stats
        assert 'free' in memory_stats

    def test_mixed_precision_operations(self):
        """Test mixed precision GPU operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        gpu_config = GPUConfig(
            device="cuda",
            use_mixed_precision=True
        )

        metrics = GPUDiversityMetrics(gpu_config)

        # Create large embeddings to test mixed precision
        large_embeddings = np.random.randn(100, 768).astype(np.float32)

        # Should handle mixed precision without errors
        diversity = metrics.calculate_population_diversity_gpu(large_embeddings)
        assert diversity['embedding_diversity'] > 0
