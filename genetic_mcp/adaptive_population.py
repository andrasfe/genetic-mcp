"""Adaptive population size management for genetic algorithm."""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .models import Idea

logger = logging.getLogger(__name__)


@dataclass
class PopulationConfig:
    """Configuration for adaptive population size management."""
    min_population: int = 5
    max_population: int = 100
    diversity_threshold: float = 0.3
    plateau_generations: int = 3
    convergence_threshold: float = 0.05
    adjustment_factor: float = 0.2  # How much to adjust population size (0-1)
    stability_window: int = 5  # Generations to consider for stability


@dataclass
class PopulationMetrics:
    """Metrics for population analysis."""
    generation: int
    population_size: int
    diversity_score: float
    average_fitness: float
    fitness_variance: float
    best_fitness: float
    convergence_rate: float
    adjustment_reason: str = ""


class AdaptivePopulationManager:
    """Manages dynamic population size adjustments based on diversity and performance metrics."""

    def __init__(self, config: PopulationConfig = None):
        """Initialize the adaptive population manager.

        Args:
            config: Configuration for population management
        """
        self.config = config or PopulationConfig()

        # Historical tracking
        self.metrics_history: deque = deque(maxlen=20)  # Keep last 20 generations
        self.population_size_history: list[int] = []
        self.adjustment_history: list[dict] = []

        # Current state tracking
        self.current_population_size: int = 10  # Default starting size
        self.plateau_counter: int = 0
        self.last_best_fitness: float = 0.0
        self.fitness_plateau_threshold: float = 0.01

        logger.info(f"Initialized AdaptivePopulationManager with config: {self.config}")

    def analyze_population(
        self,
        population: list[Idea],
        generation: int,
        diversity_metrics: dict[str, float] = None
    ) -> PopulationMetrics:
        """Analyze current population and calculate metrics.

        Args:
            population: Current population of ideas
            generation: Current generation number
            diversity_metrics: Pre-calculated diversity metrics from DiversityManager

        Returns:
            PopulationMetrics object with current analysis
        """
        if not population:
            return PopulationMetrics(
                generation=generation,
                population_size=0,
                diversity_score=0.0,
                average_fitness=0.0,
                fitness_variance=0.0,
                best_fitness=0.0,
                convergence_rate=0.0
            )

        # Extract fitness values
        fitness_values = [idea.fitness for idea in population]

        # Calculate basic fitness metrics
        average_fitness = np.mean(fitness_values)
        fitness_variance = np.var(fitness_values)
        best_fitness = max(fitness_values)

        # Use provided diversity metrics or calculate simple diversity
        if diversity_metrics:
            diversity_score = diversity_metrics.get("simpson_diversity", 0.0)
        else:
            diversity_score = self._calculate_simple_diversity(population)

        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate(fitness_values)

        metrics = PopulationMetrics(
            generation=generation,
            population_size=len(population),
            diversity_score=diversity_score,
            average_fitness=average_fitness,
            fitness_variance=fitness_variance,
            best_fitness=best_fitness,
            convergence_rate=convergence_rate
        )

        # Store in history
        self.metrics_history.append(metrics)

        logger.debug(f"Population analysis for generation {generation}: "
                    f"size={len(population)}, diversity={diversity_score:.3f}, "
                    f"avg_fitness={average_fitness:.3f}, best_fitness={best_fitness:.3f}")

        return metrics

    def get_recommended_population_size(
        self,
        current_metrics: PopulationMetrics,
        next_generation: int
    ) -> int:
        """Recommend population size for the next generation.

        Args:
            current_metrics: Current population metrics
            next_generation: Next generation number

        Returns:
            Recommended population size for next generation
        """
        # Initialize with current size if not set
        if not hasattr(self, 'current_population_size') or self.current_population_size == 0:
            self.current_population_size = current_metrics.population_size

        # Start with current size
        recommended_size = self.current_population_size
        adjustment_reasons = []

        # Check for diversity-based adjustments
        if current_metrics.diversity_score < self.config.diversity_threshold:
            # Low diversity - increase population
            increase_factor = 1.0 + self.config.adjustment_factor
            recommended_size = int(recommended_size * increase_factor)
            adjustment_reasons.append("low_diversity")
            logger.debug(f"Low diversity ({current_metrics.diversity_score:.3f}) detected, "
                        f"increasing population size by factor {increase_factor}")

        # Check for fitness plateau
        plateau_detected = self._detect_fitness_plateau(current_metrics)
        if plateau_detected:
            # Fitness plateau - increase population for exploration
            increase_factor = 1.0 + (self.config.adjustment_factor * 0.8)
            recommended_size = int(recommended_size * increase_factor)
            adjustment_reasons.append("fitness_plateau")
            logger.debug(f"Fitness plateau detected (plateau_counter={self.plateau_counter}), "
                        f"increasing population size by factor {increase_factor}")

        # Check for rapid convergence
        if current_metrics.convergence_rate > 0.8 and current_metrics.fitness_variance < 0.1:
            # Very rapid convergence - decrease population
            decrease_factor = 1.0 - (self.config.adjustment_factor * 0.6)
            recommended_size = int(recommended_size * decrease_factor)
            adjustment_reasons.append("rapid_convergence")
            logger.debug(f"Rapid convergence detected (rate={current_metrics.convergence_rate:.3f}), "
                        f"decreasing population size by factor {decrease_factor}")

        # Check for stability - if things are going well, gradually reduce population
        if (self._is_population_stable() and len(adjustment_reasons) == 0 and
            current_metrics.diversity_score > self.config.diversity_threshold * 1.5):
            # Good diversity and stable - slightly reduce population for efficiency
            decrease_factor = 1.0 - (self.config.adjustment_factor * 0.3)
            recommended_size = int(recommended_size * decrease_factor)
            adjustment_reasons.append("stable_optimization")
            logger.debug("Stable population with good diversity, slightly reducing size for efficiency")

        # Apply bounds
        recommended_size = max(self.config.min_population,
                             min(self.config.max_population, recommended_size))

        # Ensure minimum change threshold to avoid constant small adjustments
        size_change = abs(recommended_size - self.current_population_size)
        min_change_threshold = max(1, int(self.current_population_size * 0.1))  # At least 10% change

        if size_change < min_change_threshold and size_change > 0:
            # Change is too small, stick with current size
            recommended_size = self.current_population_size
            adjustment_reasons = ["insufficient_change"]

        # Log adjustment if size changed
        if recommended_size != self.current_population_size:
            adjustment = {
                "generation": next_generation,
                "from_size": self.current_population_size,
                "to_size": recommended_size,
                "reasons": adjustment_reasons,
                "diversity_score": current_metrics.diversity_score,
                "convergence_rate": current_metrics.convergence_rate,
                "fitness_variance": current_metrics.fitness_variance
            }

            self.adjustment_history.append(adjustment)

            logger.info(f"Population size adjustment for generation {next_generation}: "
                       f"{self.current_population_size} -> {recommended_size} "
                       f"(reasons: {', '.join(adjustment_reasons)})")

        # Update tracking
        self.current_population_size = recommended_size
        self.population_size_history.append(recommended_size)

        return recommended_size

    def update_session_population_size(self, session, new_size: int) -> None:
        """Update session's population size parameters.

        Args:
            session: Session object to update
            new_size: New population size
        """
        if hasattr(session, 'parameters'):
            old_size = session.parameters.population_size
            session.parameters.population_size = new_size

            # Update elitism count proportionally
            if hasattr(session.parameters, 'elitism_count'):
                # Maintain same ratio of elites
                elite_ratio = session.parameters.elitism_count / old_size
                session.parameters.elitism_count = max(1, int(new_size * elite_ratio))

            logger.debug(f"Updated session {session.id} population size: {old_size} -> {new_size}")

    def _calculate_simple_diversity(self, population: list[Idea]) -> float:
        """Calculate simple diversity metric based on content similarity.

        Args:
            population: List of ideas

        Returns:
            Diversity score between 0 and 1
        """
        if len(population) < 2:
            return 1.0

        # Simple diversity based on unique content prefixes
        unique_prefixes = set()
        for idea in population:
            # Use first 50 characters as diversity signature
            prefix = idea.content[:50].lower().strip()
            unique_prefixes.add(prefix)

        diversity = len(unique_prefixes) / len(population)
        return diversity

    def _calculate_convergence_rate(self, fitness_values: list[float]) -> float:
        """Calculate convergence rate based on fitness distribution.

        Args:
            fitness_values: List of fitness values

        Returns:
            Convergence rate between 0 and 1 (1 = fully converged)
        """
        if len(fitness_values) < 2:
            return 0.0

        # Calculate coefficient of variation (CV)
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        if mean_fitness == 0:
            return 1.0  # All fitness values are zero - fully converged

        cv = std_fitness / mean_fitness

        # Convert CV to convergence rate (higher CV = lower convergence)
        # CV of 0 = full convergence, CV of 1+ = no convergence
        convergence_rate = max(0.0, min(1.0, 1.0 - cv))

        return convergence_rate

    def _detect_fitness_plateau(self, current_metrics: PopulationMetrics) -> bool:
        """Detect if fitness has plateaued.

        Args:
            current_metrics: Current population metrics

        Returns:
            True if plateau detected
        """
        # Check if best fitness has improved significantly
        fitness_improvement = current_metrics.best_fitness - self.last_best_fitness

        if fitness_improvement < self.fitness_plateau_threshold:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0

        self.last_best_fitness = current_metrics.best_fitness

        plateau_detected = self.plateau_counter >= self.config.plateau_generations

        if plateau_detected:
            logger.debug(f"Fitness plateau detected: {self.plateau_counter} generations "
                        f"without improvement > {self.fitness_plateau_threshold}")

        return plateau_detected

    def _is_population_stable(self) -> bool:
        """Check if population has been stable recently.

        Returns:
            True if population has been stable
        """
        if len(self.metrics_history) < self.config.stability_window:
            return False

        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-self.config.stability_window:]

        # Check diversity stability
        diversity_scores = [m.diversity_score for m in recent_metrics]
        diversity_variance = np.var(diversity_scores)

        # Check fitness stability
        avg_fitness_scores = [m.average_fitness for m in recent_metrics]
        fitness_trend = np.polyfit(range(len(avg_fitness_scores)), avg_fitness_scores, 1)[0]

        # Consider stable if diversity is not too variable and fitness is improving steadily
        diversity_stable = diversity_variance < 0.1
        fitness_improving = fitness_trend > 0

        stable = diversity_stable and fitness_improving

        if stable:
            logger.debug(f"Population considered stable: diversity_var={diversity_variance:.3f}, "
                        f"fitness_trend={fitness_trend:.3f}")

        return stable

    def get_population_statistics(self) -> dict:
        """Get comprehensive statistics about population size management.

        Returns:
            Dictionary with population management statistics
        """
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = list(self.metrics_history)[-5:]  # Last 5 generations

        stats = {
            "current_population_size": self.current_population_size,
            "total_adjustments": len(self.adjustment_history),
            "plateau_counter": self.plateau_counter,
            "last_best_fitness": self.last_best_fitness,
            "config": {
                "min_population": self.config.min_population,
                "max_population": self.config.max_population,
                "diversity_threshold": self.config.diversity_threshold,
                "plateau_generations": self.config.plateau_generations
            }
        }

        if recent_metrics:
            stats["recent_metrics"] = {
                "avg_diversity": np.mean([m.diversity_score for m in recent_metrics]),
                "avg_fitness": np.mean([m.average_fitness for m in recent_metrics]),
                "avg_population_size": np.mean([m.population_size for m in recent_metrics]),
                "diversity_trend": self._calculate_trend([m.diversity_score for m in recent_metrics]),
                "fitness_trend": self._calculate_trend([m.average_fitness for m in recent_metrics])
            }

        if self.adjustment_history:
            stats["recent_adjustments"] = self.adjustment_history[-3:]  # Last 3 adjustments

        return stats

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a series of values.

        Args:
            values: List of numeric values

        Returns:
            Trend description: 'increasing', 'decreasing', or 'stable'
        """
        if len(values) < 2:
            return "stable"

        # Simple linear regression slope
        slope = np.polyfit(range(len(values)), values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def reset(self) -> None:
        """Reset the adaptive population manager state."""
        self.metrics_history.clear()
        self.population_size_history.clear()
        self.adjustment_history.clear()
        self.plateau_counter = 0
        self.last_best_fitness = 0.0
        self.current_population_size = 10

        logger.info("AdaptivePopulationManager state reset")
