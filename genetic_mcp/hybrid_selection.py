"""Hybrid selection strategies with adaptive multi-armed bandit approach."""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .logging_config import log_operation, log_performance, setup_logging
from .models import Idea

logger = setup_logging(component="hybrid_selection")


class SelectionStrategy(str, Enum):
    """Available selection strategies."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    ROULETTE_SIGMA = "roulette_sigma"  # With sigma scaling
    TRUNCATION = "truncation"
    RANK_BASED = "rank_based"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    BOLTZMANN = "boltzmann"


@dataclass
class SelectionPerformanceMetrics:
    """Performance metrics for a selection strategy."""
    strategy: SelectionStrategy
    # Selection intensity metrics
    average_fitness_improvement: float = 0.0
    fitness_variance_before: float = 0.0
    fitness_variance_after: float = 0.0
    selection_intensity: float = 0.0  # How aggressively it selects high-fitness individuals

    # Diversity metrics
    unique_individuals_selected: int = 0
    diversity_preservation_score: float = 0.0
    phenotypic_diversity_before: float = 0.0
    phenotypic_diversity_after: float = 0.0

    # Convergence metrics
    generations_to_improvement: int = 0
    convergence_speed: float = 0.0
    exploitation_score: float = 0.0  # How well it exploits good solutions
    exploration_score: float = 0.0   # How well it explores new areas

    # Usage statistics
    times_used: int = 0
    total_execution_time: float = 0.0
    success_rate: float = 0.0  # Percentage of times it led to improvement

    # Multi-armed bandit metrics
    reward_history: list[float] = None
    confidence_interval: float = 0.0
    ucb1_value: float = 0.0

    def __post_init__(self):
        if self.reward_history is None:
            self.reward_history = []


class HybridSelectionManager:
    """Manages multiple selection strategies with adaptive selection using multi-armed bandit."""

    def __init__(self,
                 strategies: list[SelectionStrategy] = None,
                 adaptation_window: int = 5,
                 exploration_constant: float = 2.0,
                 min_uses_per_strategy: int = 3):
        """Initialize hybrid selection manager.

        Args:
            strategies: List of strategies to use. If None, uses all available strategies.
            adaptation_window: Number of generations to consider for performance tracking.
            exploration_constant: UCB1 exploration parameter (higher = more exploration).
            min_uses_per_strategy: Minimum times each strategy must be used before adaptation.
        """
        self.strategies = strategies or list(SelectionStrategy)
        self.adaptation_window = adaptation_window
        self.exploration_constant = exploration_constant
        self.min_uses_per_strategy = min_uses_per_strategy

        # Performance tracking
        self.strategy_metrics: dict[SelectionStrategy, SelectionPerformanceMetrics] = {}
        self.generation_history: list[dict[str, Any]] = []
        self.current_generation = 0
        self.total_selections = 0

        # Strategy configuration
        self.strategy_configs = {
            SelectionStrategy.TOURNAMENT: {"tournament_size": 3, "adaptive_size": True},
            SelectionStrategy.ROULETTE_WHEEL: {"use_sigma_scaling": False},
            SelectionStrategy.ROULETTE_SIGMA: {"use_sigma_scaling": True, "sigma_scaling_factor": 2.0},
            SelectionStrategy.TRUNCATION: {"truncation_percentage": 0.5, "adaptive_percentage": True},
            SelectionStrategy.RANK_BASED: {"selection_pressure": 2.0, "adaptive_pressure": True},
            SelectionStrategy.STOCHASTIC_UNIVERSAL: {"pointers": 2},
            SelectionStrategy.BOLTZMANN: {"initial_temperature": 1.0, "cooling_rate": 0.95}
        }

        # Initialize metrics for all strategies
        for strategy in self.strategies:
            self.strategy_metrics[strategy] = SelectionPerformanceMetrics(strategy=strategy)

        # Current strategy and manual override
        self.current_strategy: SelectionStrategy = None
        self.manual_override_strategy: SelectionStrategy = None
        self.manual_override_generations: int = 0

        logger.info(f"Initialized HybridSelectionManager with strategies: {[s.value for s in self.strategies]}")

    def select_strategy(self, population: list[Idea], generation: int) -> SelectionStrategy:
        """Select the best strategy using UCB1 multi-armed bandit algorithm."""
        self.current_generation = generation

        # Check for manual override
        if self.manual_override_strategy and self.manual_override_generations > 0:
            self.manual_override_generations -= 1
            logger.debug(f"Using manual override strategy: {self.manual_override_strategy.value}")
            return self.manual_override_strategy

        # Early phase: try each strategy at least min_uses_per_strategy times
        underused_strategies = [
            strategy for strategy in self.strategies
            if self.strategy_metrics[strategy].times_used < self.min_uses_per_strategy
        ]

        if underused_strategies:
            strategy = random.choice(underused_strategies)
            logger.debug(f"Exploring underused strategy: {strategy.value}")
            return strategy

        # Calculate UCB1 values for all strategies
        ucb1_values = {}
        for strategy in self.strategies:
            ucb1_values[strategy] = self._calculate_ucb1_value(strategy)

        # Select strategy with highest UCB1 value
        best_strategy = max(ucb1_values.keys(), key=lambda s: ucb1_values[s])

        logger.debug(f"UCB1 values: {[(s.value, f'{v:.3f}') for s, v in ucb1_values.items()]}")
        logger.debug(f"Selected strategy: {best_strategy.value}")

        return best_strategy

    def _calculate_ucb1_value(self, strategy: SelectionStrategy) -> float:
        """Calculate UCB1 value for a strategy."""
        metrics = self.strategy_metrics[strategy]

        if metrics.times_used == 0:
            return float('inf')  # Unexplored strategies get highest priority

        # Average reward (combination of multiple performance metrics)
        avg_reward = self._calculate_average_reward(metrics)

        # Confidence interval
        confidence = self.exploration_constant * math.sqrt(
            math.log(self.total_selections + 1) / metrics.times_used
        )

        ucb1_value = avg_reward + confidence
        metrics.ucb1_value = ucb1_value
        metrics.confidence_interval = confidence

        return ucb1_value

    def _calculate_average_reward(self, metrics: SelectionPerformanceMetrics) -> float:
        """Calculate average reward for a strategy based on multiple performance criteria."""
        if not metrics.reward_history:
            return 0.0

        # Combine multiple reward components
        fitness_reward = np.mean([r for r in metrics.reward_history if 'fitness' in str(r)])
        diversity_reward = metrics.diversity_preservation_score
        convergence_reward = 1.0 / (1.0 + metrics.generations_to_improvement) if metrics.generations_to_improvement > 0 else 0.0

        # Weighted combination
        total_reward = (
            0.5 * fitness_reward +
            0.3 * diversity_reward +
            0.2 * convergence_reward
        )

        return max(0.0, min(1.0, total_reward))  # Normalize to [0, 1]

    def perform_selection(self,
                         population: list[Idea],
                         strategy: SelectionStrategy,
                         num_parents: int = 2,
                         **kwargs) -> list[Idea]:
        """Perform selection using the specified strategy."""
        start_time = time.time()

        log_operation(logger, "PERFORM_SELECTION",
                     strategy=strategy.value,
                     population_size=len(population),
                     num_parents=num_parents)

        # Pre-selection metrics
        pre_fitness = [idea.fitness for idea in population]
        pre_fitness_var = np.var(pre_fitness) if pre_fitness else 0.0
        pre_diversity = self._calculate_phenotypic_diversity(population)

        # Perform selection based on strategy
        selected_parents = []

        if strategy == SelectionStrategy.TOURNAMENT:
            selected_parents = self._tournament_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.ROULETTE_WHEEL:
            selected_parents = self._roulette_wheel_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.ROULETTE_SIGMA:
            selected_parents = self._roulette_sigma_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.TRUNCATION:
            selected_parents = self._truncation_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.RANK_BASED:
            selected_parents = self._rank_based_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.STOCHASTIC_UNIVERSAL:
            selected_parents = self._stochastic_universal_selection(population, num_parents, **kwargs)
        elif strategy == SelectionStrategy.BOLTZMANN:
            selected_parents = self._boltzmann_selection(population, num_parents, **kwargs)
        else:
            # Fallback to tournament
            selected_parents = self._tournament_selection(population, num_parents)

        # Post-selection metrics
        post_fitness = [idea.fitness for idea in selected_parents]
        post_fitness_var = np.var(post_fitness) if post_fitness else 0.0
        post_diversity = self._calculate_phenotypic_diversity(selected_parents)

        # Update performance metrics
        execution_time = time.time() - start_time
        self._update_strategy_metrics(
            strategy, population, selected_parents,
            pre_fitness_var, post_fitness_var,
            pre_diversity, post_diversity,
            execution_time
        )

        log_performance(logger, "PERFORM_SELECTION", execution_time,
                       strategy=strategy.value,
                       parents_selected=len(selected_parents),
                       avg_selected_fitness=np.mean(post_fitness) if post_fitness else 0.0)

        return selected_parents

    def _tournament_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Tournament selection with adaptive tournament size."""
        config = self.strategy_configs[SelectionStrategy.TOURNAMENT]
        tournament_size = kwargs.get('tournament_size', config['tournament_size'])

        # Adaptive tournament size based on population diversity
        if config['adaptive_size']:
            diversity = self._calculate_phenotypic_diversity(population)
            if diversity < 0.3:
                tournament_size = max(2, tournament_size - 1)  # Reduce selection pressure
            elif diversity > 0.7:
                tournament_size = min(len(population) // 2, tournament_size + 1)  # Increase pressure

        selected = []
        for _ in range(num_parents):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected

    def _roulette_wheel_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Standard roulette wheel selection."""
        fitnesses = [max(0.001, idea.fitness) for idea in population]  # Ensure positive fitness
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))

        probabilities = [f / total_fitness for f in fitnesses]

        selected = []
        for _ in range(num_parents):
            choice = np.random.choice(population, p=probabilities)
            selected.append(choice)

        return selected

    def _roulette_sigma_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Roulette wheel selection with sigma scaling to prevent premature convergence."""
        config = self.strategy_configs[SelectionStrategy.ROULETTE_SIGMA]
        sigma_factor = kwargs.get('sigma_scaling_factor', config['sigma_scaling_factor'])

        fitnesses = [idea.fitness for idea in population]

        if not fitnesses:
            return random.sample(population, min(num_parents, len(population)))

        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        # Apply sigma scaling: f'(i) = max(f(i) - (μ - c*σ), 0)
        if std_fitness > 0:
            scaled_fitnesses = [
                max(0.001, fitness - (mean_fitness - sigma_factor * std_fitness))
                for fitness in fitnesses
            ]
        else:
            scaled_fitnesses = [1.0] * len(fitnesses)  # Equal probabilities if no variance

        total_scaled = sum(scaled_fitnesses)
        if total_scaled == 0:
            return random.sample(population, min(num_parents, len(population)))

        probabilities = [f / total_scaled for f in scaled_fitnesses]

        selected = []
        for _ in range(num_parents):
            choice = np.random.choice(population, p=probabilities)
            selected.append(choice)

        return selected

    def _truncation_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Truncation selection - select from top percentage of population."""
        config = self.strategy_configs[SelectionStrategy.TRUNCATION]
        truncation_pct = kwargs.get('truncation_percentage', config['truncation_percentage'])

        # Adaptive truncation percentage based on generation and diversity
        if config['adaptive_percentage']:
            diversity = self._calculate_phenotypic_diversity(population)
            if diversity < 0.3:
                truncation_pct = min(0.8, truncation_pct + 0.1)  # Select from broader pool
            elif diversity > 0.7:
                truncation_pct = max(0.2, truncation_pct - 0.1)  # Select from narrower pool

        # Sort population by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Select top percentage
        truncation_size = max(1, int(len(population) * truncation_pct))
        truncated_pool = sorted_population[:truncation_size]

        # Randomly select from truncated pool
        selected = []
        for _ in range(num_parents):
            selected.append(random.choice(truncated_pool))

        return selected

    def _rank_based_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Rank-based selection to reduce effects of fitness scaling."""
        config = self.strategy_configs[SelectionStrategy.RANK_BASED]
        selection_pressure = kwargs.get('selection_pressure', config['selection_pressure'])

        # Adaptive selection pressure
        if config['adaptive_pressure']:
            diversity = self._calculate_phenotypic_diversity(population)
            if diversity < 0.3:
                selection_pressure = max(1.1, selection_pressure - 0.2)  # Reduce pressure
            elif diversity > 0.7:
                selection_pressure = min(2.0, selection_pressure + 0.2)  # Increase pressure

        # Sort population by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_population)

        # Calculate linear ranking probabilities
        probabilities = []
        for i in range(n):
            rank = i + 1  # Rank from 1 to n
            # Linear ranking: P(i) = (2-SP)/N + 2*rank(i)*(SP-1)/(N*(N-1))
            prob = (2 - selection_pressure) / n + 2 * rank * (selection_pressure - 1) / (n * (n - 1))
            probabilities.append(max(0.001, prob))

        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        selected = []
        for _ in range(num_parents):
            choice = np.random.choice(sorted_population, p=probabilities)
            selected.append(choice)

        return selected

    def _stochastic_universal_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Stochastic Universal Sampling for reduced selection bias."""
        config = self.strategy_configs[SelectionStrategy.STOCHASTIC_UNIVERSAL]
        num_pointers = min(num_parents, kwargs.get('pointers', config['pointers']))

        # Calculate cumulative fitness
        fitnesses = [max(0.001, idea.fitness) for idea in population]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))

        # Calculate cumulative probabilities
        cumulative_probs = []
        cumsum = 0
        for fitness in fitnesses:
            cumsum += fitness / total_fitness
            cumulative_probs.append(cumsum)

        # Generate evenly spaced pointers
        pointer_distance = 1.0 / num_pointers
        start = random.random() * pointer_distance
        pointers = [start + i * pointer_distance for i in range(num_pointers)]

        # Select individuals
        selected_indices = []
        for pointer in pointers:
            for i, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    selected_indices.append(i)
                    break

        # Ensure we have enough parents
        while len(selected_indices) < num_parents:
            selected_indices.append(random.randint(0, len(population) - 1))

        selected = [population[i] for i in selected_indices[:num_parents]]
        return selected

    def _boltzmann_selection(self, population: list[Idea], num_parents: int, **kwargs) -> list[Idea]:
        """Boltzmann selection with temperature-based probability distribution."""
        config = self.strategy_configs[SelectionStrategy.BOLTZMANN]

        # Temperature annealing: T(t) = T_0 * α^t
        initial_temp = kwargs.get('initial_temperature', config['initial_temperature'])
        cooling_rate = kwargs.get('cooling_rate', config['cooling_rate'])
        temperature = initial_temp * (cooling_rate ** self.current_generation)
        temperature = max(0.01, temperature)  # Prevent temperature from reaching 0

        # Extract and normalize fitnesses
        fitnesses = np.array([idea.fitness for idea in population])

        # Shift fitnesses to prevent negative exponents
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6

        # Scale by temperature
        scaled_fitnesses = fitnesses / temperature

        # Prevent overflow
        max_scaled = np.max(scaled_fitnesses)
        if max_scaled > 700:  # exp(700) is near float64 limit
            scaled_fitnesses = scaled_fitnesses * (700 / max_scaled)

        # Calculate Boltzmann probabilities
        exp_fitnesses = np.exp(scaled_fitnesses)
        probabilities = exp_fitnesses / np.sum(exp_fitnesses)

        # Ensure probabilities are valid
        probabilities = probabilities / np.sum(probabilities)

        selected = []
        for _ in range(num_parents):
            try:
                choice = np.random.choice(population, p=probabilities)
                selected.append(choice)
            except ValueError:
                # Fallback to random selection if probabilities are invalid
                selected.append(random.choice(population))

        return selected

    def _calculate_phenotypic_diversity(self, population: list[Idea]) -> float:
        """Calculate phenotypic diversity of population based on content similarity."""
        if len(population) <= 1:
            return 0.0

        # Simple diversity metric based on content uniqueness
        unique_prefixes = len(set(idea.content[:50] for idea in population))
        diversity = unique_prefixes / len(population)

        return diversity

    def _update_strategy_metrics(self,
                                strategy: SelectionStrategy,
                                original_population: list[Idea],
                                selected_parents: list[Idea],
                                pre_fitness_var: float,
                                post_fitness_var: float,
                                pre_diversity: float,
                                post_diversity: float,
                                execution_time: float) -> None:
        """Update performance metrics for a strategy."""
        metrics = self.strategy_metrics[strategy]
        metrics.times_used += 1
        metrics.total_execution_time += execution_time
        self.total_selections += 1

        # Calculate selection intensity
        avg_original_fitness = np.mean([idea.fitness for idea in original_population])
        avg_selected_fitness = np.mean([idea.fitness for idea in selected_parents])
        fitness_improvement = avg_selected_fitness - avg_original_fitness

        # Update metrics
        metrics.average_fitness_improvement = (
            (metrics.average_fitness_improvement * (metrics.times_used - 1) + fitness_improvement) /
            metrics.times_used
        )

        metrics.fitness_variance_before = pre_fitness_var
        metrics.fitness_variance_after = post_fitness_var
        metrics.selection_intensity = avg_selected_fitness / max(0.001, avg_original_fitness)

        metrics.unique_individuals_selected = len(set(idea.id for idea in selected_parents))
        metrics.diversity_preservation_score = post_diversity / max(0.001, pre_diversity)
        metrics.phenotypic_diversity_before = pre_diversity
        metrics.phenotypic_diversity_after = post_diversity

        # Calculate exploration vs exploitation scores
        fitness_range = max([idea.fitness for idea in original_population]) - min([idea.fitness for idea in original_population])
        if fitness_range > 0:
            selected_fitness_spread = (
                max([idea.fitness for idea in selected_parents]) -
                min([idea.fitness for idea in selected_parents])
            )
            metrics.exploration_score = selected_fitness_spread / fitness_range
        else:
            metrics.exploration_score = 0.0

        metrics.exploitation_score = min(1.0, fitness_improvement / max(0.001, avg_original_fitness))

        # Update reward history for UCB1
        reward = (
            0.4 * min(1.0, max(0.0, fitness_improvement)) +  # Normalized fitness improvement
            0.3 * metrics.diversity_preservation_score +      # Diversity preservation
            0.3 * metrics.exploitation_score                  # Exploitation success
        )

        metrics.reward_history.append(reward)

        # Keep only recent rewards (sliding window)
        if len(metrics.reward_history) > self.adaptation_window * 2:
            metrics.reward_history = metrics.reward_history[-self.adaptation_window:]

        # Update success rate
        successful_selections = sum(1 for r in metrics.reward_history if r > 0.5)
        metrics.success_rate = successful_selections / len(metrics.reward_history) if metrics.reward_history else 0.0

    def get_strategy_recommendations(self, population: list[Idea]) -> dict[str, Any]:
        """Get recommendations for strategy selection based on current population state."""
        diversity = self._calculate_phenotypic_diversity(population)
        fitness_variance = np.var([idea.fitness for idea in population])

        recommendations = {
            'recommended_strategies': [],
            'population_analysis': {
                'diversity': diversity,
                'fitness_variance': fitness_variance,
                'population_size': len(population)
            },
            'strategy_performance': {}
        }

        # Analyze current population state
        if diversity < 0.3:
            # Low diversity - recommend exploratory strategies
            recommendations['recommended_strategies'].extend([
                SelectionStrategy.STOCHASTIC_UNIVERSAL,
                SelectionStrategy.RANK_BASED,
                SelectionStrategy.ROULETTE_SIGMA
            ])
        elif diversity > 0.7:
            # High diversity - recommend exploitative strategies
            recommendations['recommended_strategies'].extend([
                SelectionStrategy.TOURNAMENT,
                SelectionStrategy.TRUNCATION,
                SelectionStrategy.BOLTZMANN
            ])
        else:
            # Balanced diversity - recommend balanced strategies
            recommendations['recommended_strategies'].extend([
                SelectionStrategy.TOURNAMENT,
                SelectionStrategy.RANK_BASED,
                SelectionStrategy.ROULETTE_WHEEL
            ])

        # Add performance summary for each strategy
        for strategy, metrics in self.strategy_metrics.items():
            recommendations['strategy_performance'][strategy.value] = {
                'times_used': metrics.times_used,
                'success_rate': metrics.success_rate,
                'avg_fitness_improvement': metrics.average_fitness_improvement,
                'diversity_preservation': metrics.diversity_preservation_score,
                'ucb1_value': metrics.ucb1_value,
                'confidence_interval': metrics.confidence_interval
            }

        return recommendations

    def set_manual_override(self, strategy: SelectionStrategy, generations: int = 1) -> None:
        """Manually override strategy selection for a specified number of generations."""
        self.manual_override_strategy = strategy
        self.manual_override_generations = generations
        logger.info(f"Manual override set: {strategy.value} for {generations} generations")

    def clear_manual_override(self) -> None:
        """Clear any manual strategy override."""
        self.manual_override_strategy = None
        self.manual_override_generations = 0
        logger.info("Manual override cleared")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report for all strategies."""
        report = {
            'total_selections': self.total_selections,
            'current_generation': self.current_generation,
            'strategy_performance': {},
            'rankings': {
                'by_success_rate': [],
                'by_fitness_improvement': [],
                'by_diversity_preservation': [],
                'by_ucb1_value': []
            }
        }

        # Collect strategy performance
        strategies_data = []
        for strategy, metrics in self.strategy_metrics.items():
            strategy_data = {
                'strategy': strategy.value,
                'times_used': metrics.times_used,
                'success_rate': metrics.success_rate,
                'avg_fitness_improvement': metrics.average_fitness_improvement,
                'diversity_preservation_score': metrics.diversity_preservation_score,
                'selection_intensity': metrics.selection_intensity,
                'exploration_score': metrics.exploration_score,
                'exploitation_score': metrics.exploitation_score,
                'ucb1_value': metrics.ucb1_value,
                'confidence_interval': metrics.confidence_interval,
                'avg_execution_time': (
                    metrics.total_execution_time / metrics.times_used
                    if metrics.times_used > 0 else 0.0
                )
            }
            strategies_data.append(strategy_data)
            report['strategy_performance'][strategy.value] = strategy_data

        # Create rankings
        if strategies_data:
            report['rankings']['by_success_rate'] = sorted(
                strategies_data, key=lambda x: x['success_rate'], reverse=True
            )
            report['rankings']['by_fitness_improvement'] = sorted(
                strategies_data, key=lambda x: x['avg_fitness_improvement'], reverse=True
            )
            report['rankings']['by_diversity_preservation'] = sorted(
                strategies_data, key=lambda x: x['diversity_preservation_score'], reverse=True
            )
            report['rankings']['by_ucb1_value'] = sorted(
                strategies_data, key=lambda x: x['ucb1_value'], reverse=True
            )

        return report

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        for strategy in self.strategies:
            self.strategy_metrics[strategy] = SelectionPerformanceMetrics(strategy=strategy)

        self.generation_history.clear()
        self.current_generation = 0
        self.total_selections = 0

        logger.info("All strategy metrics have been reset")

    def update_strategy_config(self, strategy: SelectionStrategy, config_updates: dict[str, Any]) -> None:
        """Update configuration for a specific strategy."""
        if strategy in self.strategy_configs:
            self.strategy_configs[strategy].update(config_updates)
            logger.info(f"Updated config for {strategy.value}: {config_updates}")
        else:
            logger.warning(f"Unknown strategy: {strategy.value}")

    def get_adaptive_selection_pressure(self, population: list[Idea], generation: int) -> float:
        """Calculate adaptive selection pressure based on population state and generation."""
        diversity = self._calculate_phenotypic_diversity(population)
        fitness_variance = np.var([idea.fitness for idea in population])

        # Base pressure starts high and decreases over generations
        base_pressure = max(0.5, 2.0 - (generation * 0.1))

        # Adjust based on diversity
        if diversity < 0.3:
            # Low diversity - reduce pressure to encourage exploration
            pressure_adjustment = -0.5
        elif diversity > 0.7:
            # High diversity - increase pressure to encourage exploitation
            pressure_adjustment = 0.5
        else:
            pressure_adjustment = 0.0

        # Adjust based on fitness variance
        if fitness_variance < 0.01:
            # Low variance - reduce pressure to maintain diversity
            pressure_adjustment -= 0.3

        final_pressure = max(0.5, min(2.0, base_pressure + pressure_adjustment))

        return final_pressure
