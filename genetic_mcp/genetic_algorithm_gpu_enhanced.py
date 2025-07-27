"""Enhanced GPU-optimized genetic algorithm with advanced selection and diversity preservation.

This module integrates GPU-accelerated selection strategies, diversity metrics,
and multi-population management for high-performance genetic algorithms.
"""

import logging
from typing import Any

import numpy as np

from .gpu_accelerated import GPUConfig, create_gpu_accelerated_components
from .gpu_diversity_metrics import GPUDiversityMetrics
from .gpu_selection_optimized import GPUOptimizedSelection
from .llm_client import LLMClient
from .models import FitnessWeights, GeneticParameters, Idea

logger = logging.getLogger(__name__)


class AdvancedGeneticParameters(GeneticParameters):
    """Extended parameters for advanced genetic algorithms."""
    # Fitness weights
    fitness_weights: FitnessWeights = FitnessWeights()

    # Selection parameters
    selection_method: str = "tournament"  # tournament, boltzmann, sus, rank
    tournament_size: int = 3
    selection_pressure: float = 0.9
    temperature_initial: float = 1.0
    temperature_decay: float = 0.95

    # Diversity preservation
    use_fitness_sharing: bool = False
    sigma_share: float = 0.1
    sharing_alpha: float = 1.0
    use_crowding: bool = False
    diversity_weight: float = 0.3

    # Multi-objective optimization
    use_pareto_ranking: bool = False
    use_nsga2: bool = False

    # Adaptive parameters
    adaptive_mutation: bool = True
    adaptive_crossover: bool = True
    mutation_rate_min: float = 0.01
    mutation_rate_max: float = 0.5

    # Multi-population
    n_subpopulations: int = 1
    migration_rate: float = 0.1
    migration_interval: int = 5


class GPUEnhancedGeneticAlgorithm:
    """GPU-enhanced genetic algorithm with advanced features."""

    def __init__(
        self,
        parameters: AdvancedGeneticParameters = None,
        llm_client: LLMClient | None = None,
        gpu_config: GPUConfig = None
    ):
        self.parameters = parameters or AdvancedGeneticParameters()
        self.llm_client = llm_client
        self.gpu_config = gpu_config or GPUConfig()

        # Initialize GPU components only if PyTorch is available
        try:
            self.gpu_components = create_gpu_accelerated_components(self.gpu_config)
            self.gpu_available = True
        except RuntimeError as e:
            logger.warning(f"GPU acceleration unavailable: {e}")
            self.gpu_components = None
            self.gpu_available = False

        self.selection_optimizer = GPUOptimizedSelection(self.gpu_config)
        self.diversity_metrics = GPUDiversityMetrics(self.gpu_config)

        # Import fitness evaluator dynamically
        if self.gpu_available:
            try:
                from .fitness_gpu import GPUOptimizedFitnessEvaluator
                self.fitness_evaluator = GPUOptimizedFitnessEvaluator(
                    weights=self.parameters.fitness_weights,
                    gpu_config=self.gpu_config
                )
            except (ImportError, RuntimeError) as e:
                logger.warning(f"GPU fitness evaluator unavailable: {e}")
                from .fitness import FitnessEvaluator
                self.fitness_evaluator = FitnessEvaluator(
                    weights=self.parameters.fitness_weights
                )
        else:
            from .fitness import FitnessEvaluator
            self.fitness_evaluator = FitnessEvaluator(
                weights=self.parameters.fitness_weights
            )

        # Evolution tracking
        self.generation = 0
        self.temperature = self.parameters.temperature_initial
        self.population_history = []
        self.fitness_history = []
        self.diversity_history = []

        # Multi-population support
        self.subpopulations = []
        self.migration_counter = 0

        logger.info(f"Initialized GPU-enhanced genetic algorithm on {self.gpu_config.device}")

    async def evolve_population(
        self,
        initial_population: list[Idea],
        target_prompt: str,
        generations: int,
        callbacks: dict[str, Any] | None = None
    ) -> list[Idea]:
        """Evolve population for specified generations with GPU acceleration."""
        population = initial_population

        # Split into subpopulations if configured
        if self.parameters.n_subpopulations > 1:
            self.subpopulations = self._split_population(population)
        else:
            self.subpopulations = [population]

        for gen in range(generations):
            self.generation = gen
            logger.info(f"Generation {gen + 1}/{generations}")

            # Process each subpopulation
            evolved_subpops = []
            for subpop_idx, subpop in enumerate(self.subpopulations):
                # Evaluate fitness
                if hasattr(self.fitness_evaluator, 'evaluate_population_async'):
                    await self.fitness_evaluator.evaluate_population_async(
                        subpop, target_prompt
                    )
                else:
                    # Fallback to sync method for regular FitnessEvaluator
                    # Need to get target embedding first
                    if self.gpu_available and self.gpu_components:
                        embeddings_dict = await self.gpu_components["embedding_generator"].generate_embeddings(
                            [target_prompt], ['target']
                        )
                        target_embedding = embeddings_dict['target'].tolist()
                    else:
                        # Dummy embedding for testing
                        target_embedding = np.random.randn(self.gpu_config.embedding_dim).tolist()

                    self.fitness_evaluator.evaluate_population(subpop, target_embedding)

                # Apply fitness sharing if enabled
                if self.parameters.use_fitness_sharing:
                    await self._apply_fitness_sharing(subpop)

                # Calculate diversity metrics
                embeddings = await self._get_population_embeddings(subpop)
                fitness_scores = np.array([idea.fitness for idea in subpop])

                diversity_metrics = self.diversity_metrics.calculate_population_diversity_gpu(
                    embeddings, fitness_scores
                )

                # Log metrics
                logger.info(f"Subpop {subpop_idx}: Diversity={diversity_metrics['embedding_diversity']:.3f}, "
                          f"Best fitness={fitness_scores.max():.3f}")

                # Select and create next generation
                next_gen = await self._create_next_generation_advanced(
                    subpop, fitness_scores, embeddings, target_prompt
                )

                evolved_subpops.append(next_gen)

            # Migration between subpopulations
            if self.parameters.n_subpopulations > 1:
                self.migration_counter += 1
                if self.migration_counter >= self.parameters.migration_interval:
                    evolved_subpops = self._perform_migration(evolved_subpops)
                    self.migration_counter = 0

            # Update subpopulations
            self.subpopulations = evolved_subpops

            # Combine for tracking
            population = [idea for subpop in self.subpopulations for idea in subpop]

            # Track history
            all_embeddings = await self._get_population_embeddings(population)
            all_fitness = np.array([idea.fitness for idea in population])
            self.population_history.append(all_embeddings)
            self.fitness_history.append(all_fitness)

            # Adaptive parameter updates
            self._update_adaptive_parameters(diversity_metrics)

            # Callbacks
            if callbacks and 'on_generation' in callbacks:
                await callbacks['on_generation'](gen, population, diversity_metrics)

        # Final population
        final_population = [idea for subpop in self.subpopulations for idea in subpop]

        # Select diverse top solutions
        if len(final_population) > self.parameters.population_size:
            final_population = await self.fitness_evaluator.find_diverse_subset(
                final_population,
                self.parameters.population_size,
                self.parameters.diversity_weight
            )

        return final_population

    async def _create_next_generation_advanced(
        self,
        population: list[Idea],
        fitness_scores: np.ndarray,
        embeddings: np.ndarray,
        target_prompt: str
    ) -> list[Idea]:
        """Create next generation using advanced GPU-accelerated strategies."""
        new_population = []
        len(population)

        # Handle elitism
        elite_count = self._adaptive_elitism_count(fitness_scores)
        if elite_count > 0:
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx, elite_idx in enumerate(elite_indices):
                elite_idea = population[elite_idx]
                new_idea = Idea(
                    id=f"gen{self.generation}_elite{idx}",
                    content=elite_idea.content,
                    generation=self.generation,
                    parent_ids=[elite_idea.id],
                    metadata={"elite": True, "preserved_fitness": elite_idea.fitness}
                )
                new_population.append(new_idea)

        # Calculate remaining slots
        remaining_slots = self.parameters.population_size - len(new_population)

        # Prepare selection based on method
        if self.parameters.use_crowding:
            # Calculate crowding distances
            objective_values = self._get_objective_values(population)
            crowding_distances = self.selection_optimizer.crowding_distance_gpu(
                objective_values
            )
            # Combine fitness and crowding distance
            selection_scores = fitness_scores * (1 + crowding_distances)
        else:
            selection_scores = fitness_scores

        # Batch parent selection using configured method
        parent_indices = await self._select_parents_batch_gpu(
            selection_scores, embeddings, remaining_slots
        )

        # Create offspring
        offspring_count = 0
        for i in range(0, len(parent_indices), 2):
            if i + 1 < len(parent_indices) and len(new_population) < self.parameters.population_size:
                parent1 = population[parent_indices[i]]
                parent2 = population[parent_indices[i + 1]]

                # Crossover
                if self.llm_client and np.random.random() < self.parameters.crossover_rate:
                    offspring1_content, offspring2_content = await self._semantic_crossover_gpu(
                        parent1, parent2
                    )
                else:
                    offspring1_content, offspring2_content = self._simple_crossover(
                        parent1.content, parent2.content
                    )

                # Mutation with adaptive rate
                mutation_rate = self._get_adaptive_mutation_rate(fitness_scores)

                if np.random.random() < mutation_rate:
                    offspring1_content = await self._mutate_content(offspring1_content)
                if np.random.random() < mutation_rate:
                    offspring2_content = await self._mutate_content(offspring2_content)

                # Create offspring ideas
                offspring1 = Idea(
                    id=f"gen{self.generation}_offspring{offspring_count}",
                    content=offspring1_content,
                    generation=self.generation,
                    parent_ids=[parent1.id, parent2.id],
                    metadata={"selection_method": self.parameters.selection_method}
                )
                new_population.append(offspring1)
                offspring_count += 1

                if len(new_population) < self.parameters.population_size:
                    offspring2 = Idea(
                        id=f"gen{self.generation}_offspring{offspring_count}",
                        content=offspring2_content,
                        generation=self.generation,
                        parent_ids=[parent1.id, parent2.id],
                        metadata={"selection_method": self.parameters.selection_method}
                    )
                    new_population.append(offspring2)
                    offspring_count += 1

        return new_population[:self.parameters.population_size]

    async def _select_parents_batch_gpu(
        self,
        selection_scores: np.ndarray,
        embeddings: np.ndarray,
        num_parents: int
    ) -> np.ndarray:
        """Select parents using GPU-accelerated methods."""
        method = self.parameters.selection_method

        if method == "boltzmann":
            return self.selection_optimizer.boltzmann_selection_batch(
                selection_scores, self.temperature, num_parents
            )
        elif method == "sus":
            return self.selection_optimizer.stochastic_universal_sampling_gpu(
                selection_scores, num_parents
            )
        elif method == "rank":
            return self.selection_optimizer.rank_based_selection_gpu(
                selection_scores, num_parents, self.parameters.selection_pressure
            )
        elif method == "diversity":
            return self.selection_optimizer.diversity_preservation_selection_gpu(
                selection_scores, embeddings, num_parents, self.parameters.diversity_weight
            )
        else:  # tournament
            return self.selection_optimizer.batch_tournament_selection_advanced_gpu(
                selection_scores, num_parents,
                self.parameters.tournament_size,
                self.parameters.selection_pressure
            )

    async def _apply_fitness_sharing(self, population: list[Idea]) -> None:
        """Apply fitness sharing to preserve diversity."""
        embeddings = await self._get_population_embeddings(population)
        fitness_scores = np.array([idea.fitness for idea in population])

        shared_fitness = self.selection_optimizer.fitness_sharing_gpu(
            fitness_scores, embeddings,
            self.parameters.sigma_share,
            self.parameters.sharing_alpha
        )

        # Update fitness scores
        for idea, shared_fit in zip(population, shared_fitness, strict=False):
            idea.metadata['raw_fitness'] = idea.fitness
            idea.fitness = float(shared_fit)

    async def _get_population_embeddings(self, population: list[Idea]) -> np.ndarray:
        """Get embeddings for entire population."""
        texts = [idea.content for idea in population]
        ids = [idea.id for idea in population]

        if self.gpu_available and self.gpu_components:
            embeddings_dict = await self.gpu_components["embedding_generator"].generate_embeddings(
                texts, ids
            )
        else:
            # Fallback to dummy embeddings for testing
            embeddings_dict = {id_: np.random.randn(self.gpu_config.embedding_dim) for id_ in ids}

        return np.vstack([embeddings_dict[id_] for id_ in ids])

    def _get_objective_values(self, population: list[Idea]) -> np.ndarray:
        """Extract objective values for multi-objective optimization."""
        objectives = []
        for idea in population:
            obj_values = [
                idea.scores.get('relevance', 0),
                idea.scores.get('novelty', 0),
                idea.scores.get('feasibility', 0)
            ]
            objectives.append(obj_values)
        return np.array(objectives)

    def _adaptive_elitism_count(self, fitness_scores: np.ndarray) -> int:
        """Adaptively determine elitism count based on population state."""
        base_elite = self.parameters.elitism_count

        # Calculate fitness statistics
        fitness_std = np.std(fitness_scores)
        fitness_range = np.max(fitness_scores) - np.min(fitness_scores)

        # Increase elitism if population is converging well
        if fitness_std < 0.1 or fitness_range < 0.2:
            return min(base_elite + 2, len(fitness_scores) // 4)

        # Decrease elitism if population is too diverse
        if fitness_std > 0.3:
            return max(1, base_elite - 1)

        return base_elite

    def _get_adaptive_mutation_rate(self, fitness_scores: np.ndarray) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        if not self.parameters.adaptive_mutation:
            return self.parameters.mutation_rate

        # Calculate convergence metrics
        fitness_std = np.std(fitness_scores)

        # High mutation if converged, low if diverse
        if fitness_std < 0.05:
            rate = self.parameters.mutation_rate_max
        elif fitness_std > 0.3:
            rate = self.parameters.mutation_rate_min
        else:
            # Linear interpolation
            rate = (self.parameters.mutation_rate_max -
                   (fitness_std - 0.05) / 0.25 *
                   (self.parameters.mutation_rate_max - self.parameters.mutation_rate_min))

        # Generation-based adjustment
        generation_factor = 1 + (self.generation * 0.02)
        rate = min(rate * generation_factor, self.parameters.mutation_rate_max)

        return rate

    def _update_adaptive_parameters(self, diversity_metrics: dict[str, float]) -> None:
        """Update adaptive parameters based on evolution state."""
        # Temperature annealing for Boltzmann selection
        if self.parameters.selection_method == "boltzmann":
            self.temperature *= self.parameters.temperature_decay
            self.temperature = max(0.1, self.temperature)

        # Adjust selection pressure based on diversity
        if diversity_metrics.get('embedding_diversity', 0) < 0.1:
            # Low diversity, reduce selection pressure
            self.parameters.selection_pressure = max(0.5, self.parameters.selection_pressure - 0.05)
        elif diversity_metrics.get('embedding_diversity', 0) > 0.5:
            # High diversity, increase selection pressure
            self.parameters.selection_pressure = min(1.0, self.parameters.selection_pressure + 0.05)

    def _split_population(self, population: list[Idea]) -> list[list[Idea]]:
        """Split population into subpopulations."""
        n_subpops = self.parameters.n_subpopulations
        subpop_size = len(population) // n_subpops

        # Shuffle for random distribution
        shuffled = population.copy()
        np.random.shuffle(shuffled)

        subpopulations = []
        for i in range(n_subpops):
            start_idx = i * subpop_size
            end_idx = start_idx + subpop_size if i < n_subpops - 1 else len(shuffled)
            subpopulations.append(shuffled[start_idx:end_idx])

        return subpopulations

    def _perform_migration(self, subpopulations: list[list[Idea]]) -> list[list[Idea]]:
        """Perform migration between subpopulations."""
        n_subpops = len(subpopulations)
        if n_subpops < 2:
            return subpopulations

        migration_count = max(1, int(len(subpopulations[0]) * self.parameters.migration_rate))

        # Ring topology migration
        migrated_subpops = [subpop.copy() for subpop in subpopulations]

        for i in range(n_subpops):
            source_idx = i
            target_idx = (i + 1) % n_subpops

            # Select best individuals for migration
            source_pop = subpopulations[source_idx]
            fitness_scores = [idea.fitness for idea in source_pop]
            best_indices = np.argsort(fitness_scores)[-migration_count:]

            # Replace worst individuals in target
            target_pop = migrated_subpops[target_idx]
            target_fitness = [idea.fitness for idea in target_pop]
            worst_indices = np.argsort(target_fitness)[:migration_count]

            # Perform migration
            for j, (best_idx, worst_idx) in enumerate(zip(best_indices, worst_indices, strict=False)):
                migrant = source_pop[best_idx]
                # Create new idea with migration metadata
                migrated_idea = Idea(
                    id=f"gen{self.generation}_migrant{i}_{j}",
                    content=migrant.content,
                    generation=self.generation,
                    parent_ids=[migrant.id],
                    metadata={"migrated_from": source_idx, "migrated_to": target_idx}
                )
                target_pop[worst_idx] = migrated_idea

        return migrated_subpops

    async def _semantic_crossover_gpu(self, parent1: Idea, parent2: Idea) -> tuple[str, str]:
        """GPU-accelerated semantic crossover using embeddings."""
        # This is still handled by LLM, but we can use embeddings to guide
        if self.llm_client:
            prompt = f"""
            Perform intelligent crossover between these two ideas:

            Parent 1: {parent1.content}
            Parent 2: {parent2.content}

            Create two offspring that combine elements from both parents creatively.

            OFFSPRING_1: [combine mainly parent1 with elements from parent2]
            OFFSPRING_2: [combine mainly parent2 with elements from parent1]
            """

            try:
                response = await self.llm_client.generate(prompt, temperature=0.7)
                # Parse response
                lines = response.strip().split('\n')
                offspring1 = offspring2 = None

                for line in lines:
                    if line.startswith("OFFSPRING_1:"):
                        offspring1 = line.replace("OFFSPRING_1:", "").strip()
                    elif line.startswith("OFFSPRING_2:"):
                        offspring2 = line.replace("OFFSPRING_2:", "").strip()

                if offspring1 and offspring2:
                    return offspring1, offspring2
            except Exception as e:
                logger.error(f"Semantic crossover failed: {e}")

        # Fallback to simple crossover
        return self._simple_crossover(parent1.content, parent2.content)

    def _simple_crossover(self, content1: str, content2: str) -> tuple[str, str]:
        """Simple crossover implementation."""
        sentences1 = content1.split('.')
        sentences2 = content2.split('.')

        if len(sentences1) > 1 and len(sentences2) > 1:
            mid1 = len(sentences1) // 2
            mid2 = len(sentences2) // 2

            offspring1 = '. '.join(sentences1[:mid1] + sentences2[mid2:])
            offspring2 = '. '.join(sentences2[:mid2] + sentences1[mid1:])

            return offspring1.strip() + '.', offspring2.strip() + '.'

        return content1, content2

    async def _mutate_content(self, content: str) -> str:
        """Mutate content using LLM or fallback methods."""
        if self.llm_client and np.random.random() < 0.7:  # 70% chance of LLM mutation
            try:
                prompt = f"Slightly modify this idea with a creative twist: {content}"
                response = await self.llm_client.generate(prompt, temperature=0.8, max_tokens=200)
                return response.strip()
            except Exception as e:
                logger.error(f"LLM mutation failed: {e}")

        # Fallback mutations
        mutations = [
            f"{content} This could be enhanced with AI/ML techniques.",
            f"{content} Consider implementing this in a distributed manner.",
            f"{content} Real-time processing would add significant value.",
            f"Building on this concept, {content.lower()}",
            f"{content} This approach offers excellent scalability."
        ]

        return np.random.choice(mutations)

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of evolution progress."""
        if not self.fitness_history:
            return {}

        # Calculate convergence metrics
        convergence_metrics = self.diversity_metrics.calculate_convergence_metrics_gpu(
            self.population_history,
            self.fitness_history
        )

        # Overall statistics
        final_fitness = self.fitness_history[-1]
        initial_fitness = self.fitness_history[0]

        return {
            "generations": self.generation,
            "final_best_fitness": float(np.max(final_fitness)),
            "final_avg_fitness": float(np.mean(final_fitness)),
            "fitness_improvement": float(np.max(final_fitness) - np.max(initial_fitness)),
            "convergence_metrics": convergence_metrics,
            "selection_method": self.parameters.selection_method,
            "used_gpu": self.gpu_config.device != "cpu",
            "n_subpopulations": self.parameters.n_subpopulations
        }

    def cleanup(self) -> None:
        """Clean up GPU resources."""
        if self.gpu_available and self.gpu_components:
            self.gpu_components["batch_processor"].clear_caches()
        self.selection_optimizer.memory_manager.clear_cache()
        self.diversity_metrics.memory_manager.clear_cache()
