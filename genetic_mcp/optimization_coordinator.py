"""Optimization coordinator for advanced genetic algorithm execution."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .diversity_manager import DiversityManager
from .fitness_enhanced import EnhancedFitnessEvaluator
from .genetic_algorithm_optimized import OptimizedGeneticAlgorithm
from .llm_client import LLMClient
from .models import FitnessWeights, GeneticParameters, Idea, Session

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization coordinator."""
    use_adaptive_parameters: bool = True
    use_diversity_preservation: bool = True
    use_pareto_optimization: bool = True
    use_llm_operators: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 3
    diversity_threshold: float = 0.3
    target_species: int = 5
    selection_strategy: str = "adaptive"  # adaptive, tournament, boltzmann, etc.


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    avg_fitness: float
    diversity_metrics: dict[str, float]
    species_count: int
    computation_time: float
    selection_method: str
    mutation_rate: float
    crossover_rate: float


class OptimizationCoordinator:
    """Coordinates advanced genetic algorithm optimization."""

    def __init__(
        self,
        parameters: GeneticParameters,
        fitness_weights: FitnessWeights,
        llm_client: LLMClient | None = None,
        config: OptimizationConfig | None = None
    ):
        self.parameters = parameters
        self.fitness_weights = fitness_weights
        self.llm_client = llm_client
        self.config = config or OptimizationConfig()

        # Initialize components
        self.genetic_algorithm = OptimizedGeneticAlgorithm(
            parameters=parameters,
            llm_client=llm_client if config.use_llm_operators else None
        )

        self.fitness_evaluator = EnhancedFitnessEvaluator(
            weights=fitness_weights,
            llm_client=llm_client,
            use_pareto=config.use_pareto_optimization
        )

        self.diversity_manager = DiversityManager()

        # Tracking
        self.generation_history: list[GenerationStats] = []
        self.best_ideas_history: list[Idea] = []
        self.convergence_detected = False
        self.stagnation_counter = 0

    async def run_evolution(
        self,
        initial_population: list[Idea],
        target_prompt: str,
        target_embedding: np.ndarray,
        session: Session
    ) -> tuple[list[Idea], dict[str, Any]]:
        """Run the complete evolution process with all optimizations."""
        logger.info(f"Starting optimized evolution for session {session.id}")

        current_population = initial_population
        embeddings_cache = {}

        # Pre-compute initial embeddings
        for idea in initial_population:
            embeddings_cache[idea.id] = np.random.randn(768)  # Placeholder

        evolution_metadata = {
            "total_generations": 0,
            "converged_at": None,
            "final_diversity": {},
            "optimization_actions": [],
            "performance_metrics": {}
        }

        for generation in range(self.parameters.generations):
            start_time = datetime.now()

            # Evaluate current population
            await self.fitness_evaluator.evaluate_population_enhanced(
                current_population,
                target_embedding,
                target_prompt,
                generation
            )

            # Calculate diversity metrics
            diversity_metrics = self.diversity_manager.calculate_diversity_metrics(
                current_population,
                embeddings_cache
            )

            # Apply diversity preservation if enabled
            if self.config.use_diversity_preservation:
                await self._apply_diversity_preservation(
                    current_population,
                    embeddings_cache,
                    diversity_metrics
                )

            # Track statistics
            stats = self._calculate_generation_stats(
                generation,
                current_population,
                diversity_metrics,
                (datetime.now() - start_time).total_seconds()
            )
            self.generation_history.append(stats)

            # Update best ideas
            self._update_best_ideas(current_population)

            # Check convergence and early stopping
            if self.config.use_early_stopping and self._check_convergence(current_population, diversity_metrics):
                logger.info(f"Convergence detected at generation {generation}")
                evolution_metadata["converged_at"] = generation
                break

            # Select evolution strategy based on current state
            selection_method = self._select_evolution_strategy(
                generation,
                diversity_metrics,
                stats
            )

            # Create next generation
            current_population = await self.genetic_algorithm.create_next_generation_optimized(
                current_population,
                generation + 1,
                selection_method
            )

            # Add embeddings for new population
            for idea in current_population:
                if idea.id not in embeddings_cache:
                    embeddings_cache[idea.id] = np.random.randn(768)  # Placeholder

            # Apply post-generation optimizations
            current_population = await self._apply_post_generation_optimizations(
                current_population,
                embeddings_cache,
                generation + 1
            )

            # Update session progress
            session.current_generation = generation + 1
            session.ideas = current_population

            # Log progress
            logger.info(
                f"Generation {generation + 1}: "
                f"Best fitness={stats.best_fitness:.3f}, "
                f"Diversity={stats.diversity_metrics.get('simpson_diversity', 0):.3f}, "
                f"Species={stats.species_count}"
            )

        # Final evaluation
        await self.fitness_evaluator.evaluate_population_enhanced(
            current_population,
            target_embedding,
            target_prompt,
            session.current_generation
        )

        # Compile evolution metadata
        evolution_metadata.update({
            "total_generations": session.current_generation,
            "final_diversity": self.diversity_manager.calculate_diversity_metrics(
                current_population,
                embeddings_cache
            ),
            "fitness_statistics": self.fitness_evaluator.get_fitness_statistics(current_population),
            "species_statistics": self.diversity_manager.get_species_statistics(),
            "parameter_history": self._get_parameter_history(),
            "convergence_metrics": self._get_convergence_metrics()
        })

        # Get top ideas
        top_ideas = sorted(current_population, key=lambda x: x.fitness, reverse=True)[:session.parameters.top_k]

        return top_ideas, evolution_metadata

    async def _apply_diversity_preservation(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray],
        diversity_metrics: dict[str, float]
    ) -> None:
        """Apply diversity preservation mechanisms."""
        actions = []

        # Check if diversity is low
        if diversity_metrics["simpson_diversity"] < self.config.diversity_threshold:
            # Apply fitness sharing
            self.diversity_manager.apply_fitness_sharing(population, embeddings)
            actions.append("Applied fitness sharing")

            # Increase mutation rate
            self.genetic_algorithm.adaptive_mutation_rate *= 1.2
            actions.append("Increased mutation rate")

        # Apply speciation if enabled
        if len(population) > 10:
            species_dict = self.diversity_manager.apply_speciation(population, embeddings)
            actions.append(f"Created {len(species_dict)} species")

            # Adjust niche radius
            new_radius = self.diversity_manager.adaptive_niche_radius(
                population,
                self.config.target_species
            )
            actions.append(f"Adjusted niche radius to {new_radius:.3f}")

        # Calculate crowding distances for selection
        crowding_distances = self.diversity_manager.calculate_crowding_distance(population)
        for idea in population:
            idea.metadata["crowding_distance"] = crowding_distances.get(idea.id, 0)

        if actions:
            logger.info(f"Diversity preservation: {', '.join(actions)}")

    def _select_evolution_strategy(
        self,
        generation: int,
        diversity_metrics: dict[str, float],
        stats: GenerationStats
    ) -> str:
        """Select appropriate evolution strategy based on current state."""
        if not self.config.use_adaptive_parameters:
            return "tournament"

        # Early generations: explore
        if generation < 2:
            return "boltzmann"

        # Low diversity: increase selection pressure
        if diversity_metrics["simpson_diversity"] < 0.3:
            return "sus"  # Stochastic Universal Sampling

        # Stagnation: try different strategy
        if self.stagnation_counter > 2:
            strategies = ["boltzmann", "rank", "sus"]
            return strategies[generation % len(strategies)]

        # Normal evolution
        if diversity_metrics["average_distance"] > 0.5:
            return "tournament"  # Good diversity, can use stronger selection
        else:
            return "rank"  # Moderate selection pressure

    async def _apply_post_generation_optimizations(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray],
        generation: int
    ) -> list[Idea]:
        """Apply post-generation optimizations."""
        # Local search for elite individuals
        if generation > 3 and self.llm_client:
            elite_count = max(1, len(population) // 10)
            elite = sorted(population, key=lambda x: x.fitness, reverse=True)[:elite_count]

            for elite_idea in elite:
                # Apply local refinement
                refined_content = await self._local_search_refinement(elite_idea)
                if refined_content != elite_idea.content:
                    elite_idea.content = refined_content
                    elite_idea.metadata["locally_refined"] = True

        # Apply crowding if population size exceeds limit
        if len(population) > self.parameters.population_size * 1.2:
            population = self.diversity_manager.select_diverse_subset(
                population,
                self.parameters.population_size,
                embeddings
            )

        return population

    async def _local_search_refinement(self, idea: Idea) -> str:
        """Apply local search to refine an elite idea."""
        if not self.llm_client:
            return idea.content

        refinement_prompt = f"""
        Refine and improve the following idea while maintaining its core concept:

        {idea.content}

        Focus on:
        1. Clarity and coherence
        2. Practical implementation details
        3. Addressing potential challenges

        Provide the refined version only.
        """

        try:
            refined = await self.llm_client.generate(
                refinement_prompt,
                temperature=0.3,
                max_tokens=300
            )
            return refined.strip()
        except Exception as e:
            logger.error(f"Local refinement failed: {e}")
            return idea.content

    def _check_convergence(
        self,
        population: list[Idea],
        diversity_metrics: dict[str, float]
    ) -> bool:
        """Check if evolution has converged."""
        # Check fitness convergence
        fitness_converged = self.genetic_algorithm.check_convergence(
            population,
            threshold=0.01
        )

        # Check diversity convergence
        diversity_converged = (
            diversity_metrics["simpson_diversity"] < 0.2 and
            diversity_metrics["average_distance"] < 0.1
        )

        # Update stagnation counter
        if len(self.generation_history) > 1:
            improvement = (
                self.generation_history[-1].best_fitness -
                self.generation_history[-2].best_fitness
            )
            if improvement < 0.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        # Check if converged
        converged = (
            fitness_converged or
            diversity_converged or
            self.stagnation_counter >= self.config.early_stopping_patience
        )

        if converged:
            self.convergence_detected = True

        return converged

    def _calculate_generation_stats(
        self,
        generation: int,
        population: list[Idea],
        diversity_metrics: dict[str, float],
        computation_time: float
    ) -> GenerationStats:
        """Calculate statistics for current generation."""
        fitnesses = [idea.fitness for idea in population]

        return GenerationStats(
            generation=generation,
            best_fitness=max(fitnesses) if fitnesses else 0,
            avg_fitness=np.mean(fitnesses) if fitnesses else 0,
            diversity_metrics=diversity_metrics,
            species_count=len(self.diversity_manager.species),
            computation_time=computation_time,
            selection_method=self.genetic_algorithm.metrics.generation,
            mutation_rate=self.genetic_algorithm.adaptive_mutation_rate,
            crossover_rate=self.genetic_algorithm.adaptive_crossover_rate
        )

    def _update_best_ideas(self, population: list[Idea]) -> None:
        """Update history of best ideas."""
        best_idea = max(population, key=lambda x: x.fitness)

        # Check if it's better than any in history
        if not self.best_ideas_history or best_idea.fitness > self.best_ideas_history[0].fitness:
            self.best_ideas_history.insert(0, best_idea)
            # Keep only top 10
            self.best_ideas_history = self.best_ideas_history[:10]

    def _get_parameter_history(self) -> list[dict[str, float]]:
        """Get history of parameter changes."""
        history = []

        for i, stats in enumerate(self.generation_history):
            history.append({
                "generation": i,
                "mutation_rate": stats.mutation_rate,
                "crossover_rate": stats.crossover_rate,
                "selection_method": stats.selection_method
            })

        return history

    def _get_convergence_metrics(self) -> dict[str, Any]:
        """Get convergence-related metrics."""
        if not self.generation_history:
            return {}

        # Calculate improvement rates
        improvements = []
        for i in range(1, len(self.generation_history)):
            improvement = (
                self.generation_history[i].best_fitness -
                self.generation_history[i-1].best_fitness
            )
            improvements.append(improvement)

        return {
            "converged": self.convergence_detected,
            "stagnation_generations": self.stagnation_counter,
            "average_improvement": np.mean(improvements) if improvements else 0,
            "final_best_fitness": self.generation_history[-1].best_fitness,
            "generations_to_90_percent": self._find_percentage_generation(0.9)
        }

    def _find_percentage_generation(self, percentage: float) -> int | None:
        """Find generation where fitness reached percentage of final best."""
        if not self.generation_history:
            return None

        final_best = self.generation_history[-1].best_fitness
        target = final_best * percentage

        for stats in self.generation_history:
            if stats.best_fitness >= target:
                return stats.generation

        return None

    def get_optimization_report(self) -> dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            "configuration": {
                "use_adaptive_parameters": self.config.use_adaptive_parameters,
                "use_diversity_preservation": self.config.use_diversity_preservation,
                "use_pareto_optimization": self.config.use_pareto_optimization,
                "use_llm_operators": self.config.use_llm_operators
            },
            "performance": {
                "total_generations": len(self.generation_history),
                "converged": self.convergence_detected,
                "convergence_generation": self._get_convergence_metrics().get("converged_at"),
                "total_computation_time": sum(s.computation_time for s in self.generation_history)
            },
            "fitness_progression": [
                {
                    "generation": s.generation,
                    "best": s.best_fitness,
                    "average": s.avg_fitness
                }
                for s in self.generation_history
            ],
            "diversity_progression": [
                {
                    "generation": s.generation,
                    "simpson": s.diversity_metrics.get("simpson_diversity", 0),
                    "average_distance": s.diversity_metrics.get("average_distance", 0),
                    "species_count": s.species_count
                }
                for s in self.generation_history
            ],
            "parameter_adaptation": self._get_parameter_history(),
            "final_metrics": self._get_convergence_metrics()
        }
