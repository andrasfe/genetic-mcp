"""GPU-optimized genetic algorithm operations with async support."""

import asyncio
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .gpu_accelerated import (
    GPUConfig,
    create_gpu_accelerated_components,
)
from .models import GeneticParameters, Idea

logger = logging.getLogger(__name__)


class GPUOptimizedGeneticAlgorithm:
    """GPU-optimized genetic algorithm with parallel operations."""

    def __init__(
        self,
        parameters: GeneticParameters | None = None,
        gpu_config: GPUConfig | None = None
    ):
        self.parameters = parameters or GeneticParameters()
        self.gpu_config = gpu_config or GPUConfig()

        # Initialize GPU components
        self.gpu_components = create_gpu_accelerated_components(self.gpu_config)
        self.genetic_ops = self.gpu_components["genetic_ops"]

        # Thread pool for CPU-bound text operations
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Async lock for thread safety
        self._processing_lock = asyncio.Lock()

        logger.info(f"Initialized GPU-optimized genetic algorithm on {self.gpu_config.device}")

    async def create_next_generation_gpu(
        self,
        population: list[Idea],
        fitness_scores: np.ndarray,
        generation: int,
        target_prompt: str
    ) -> list[Idea]:
        """Create next generation using GPU-accelerated operations."""
        async with self._processing_lock:
            new_population = []

            # Handle elitism
            if self.parameters.elitism_count > 0:
                # Get elite indices
                elite_indices = np.argsort(fitness_scores)[-self.parameters.elitism_count:]

                # Add elite to new population
                for idx, elite_idx in enumerate(elite_indices):
                    elite_idea = population[elite_idx]
                    new_idea = Idea(
                        id=f"gen{generation}_elite{idx}",
                        content=elite_idea.content,
                        generation=generation,
                        parent_ids=[elite_idea.id],
                        metadata={"elite": True}
                    )
                    new_population.append(new_idea)

            # Calculate remaining slots
            remaining_slots = self.parameters.population_size - len(new_population)

            # Parallel parent selection
            parent_pairs = await self._select_parent_pairs_batch(
                population,
                fitness_scores,
                remaining_slots
            )

            # Parallel crossover and mutation
            offspring = await self._create_offspring_batch(
                population,
                parent_pairs,
                generation
            )

            new_population.extend(offspring[:remaining_slots])

            return new_population

    async def _select_parent_pairs_batch(
        self,
        population: list[Idea],
        fitness_scores: np.ndarray,
        num_pairs: int
    ) -> list[tuple[int, int]]:
        """Select parent pairs in batch using GPU."""
        # Tournament selection for all parents at once
        num_selections = num_pairs * 2
        selected_indices = self.genetic_ops.tournament_selection_batch(
            fitness_scores,
            num_selections,
            tournament_size=3
        )

        # Pair up selections
        parent_pairs = []
        for i in range(0, num_selections, 2):
            if i + 1 < num_selections:
                parent_pairs.append((selected_indices[i], selected_indices[i + 1]))

        return parent_pairs

    async def _create_offspring_batch(
        self,
        population: list[Idea],
        parent_pairs: list[tuple[int, int]],
        generation: int
    ) -> list[Idea]:
        """Create offspring in batch with parallel processing."""
        # Extract content and calculate sequence lengths
        contents = [idea.content for idea in population]
        sequence_lengths = np.array([len(self._extract_sentences(c)) for c in contents])

        # Generate crossover decisions and points
        crossover_info = self.genetic_ops.parallel_crossover_indices(
            parent_pairs,
            sequence_lengths,
            self.parameters.crossover_rate
        )

        # Generate mutation mask
        mutation_mask = self.genetic_ops.batch_mutation_mask(
            len(parent_pairs) * 2,  # Two offspring per pair
            self.parameters.mutation_rate
        )

        # Process crossovers and mutations in parallel
        offspring_tasks = []
        for i, ((p1_idx, p2_idx), (do_crossover, point1, point2)) in enumerate(
            zip(parent_pairs, crossover_info, strict=False)
        ):
            task = self._create_offspring_pair(
                population[p1_idx],
                population[p2_idx],
                generation,
                i,
                do_crossover,
                point1,
                point2,
                mutation_mask[i*2:(i+1)*2]
            )
            offspring_tasks.append(task)

        # Execute all tasks
        offspring_pairs = await asyncio.gather(*offspring_tasks)

        # Flatten results
        offspring = []
        for pair in offspring_pairs:
            offspring.extend(pair)

        return offspring

    async def _create_offspring_pair(
        self,
        parent1: Idea,
        parent2: Idea,
        generation: int,
        pair_idx: int,
        do_crossover: bool,
        point1: int,
        point2: int,
        mutation_flags: np.ndarray
    ) -> list[Idea]:
        """Create a pair of offspring from parents."""
        # Run crossover in thread pool
        loop = asyncio.get_event_loop()

        if do_crossover:
            offspring_contents = await loop.run_in_executor(
                self.executor,
                self._crossover_sync,
                parent1.content,
                parent2.content,
                point1,
                point2
            )
        else:
            offspring_contents = (parent1.content, parent2.content)

        # Apply mutations
        offspring = []
        for i, (content, should_mutate) in enumerate(zip(offspring_contents, mutation_flags, strict=False)):
            if should_mutate:
                content = await loop.run_in_executor(
                    self.executor,
                    self._mutate_sync,
                    content
                )

            idea = Idea(
                id=f"gen{generation}_offspring{pair_idx*2+i}",
                content=content,
                generation=generation,
                parent_ids=[parent1.id, parent2.id]
            )
            offspring.append(idea)

        return offspring

    def _crossover_sync(
        self,
        content1: str,
        content2: str,
        point1: int,
        point2: int
    ) -> tuple[str, str]:
        """Synchronous crossover operation."""
        sentences1 = self._extract_sentences(content1)
        sentences2 = self._extract_sentences(content2)

        if not sentences1 or not sentences2:
            return content1, content2

        # Ensure points are within bounds
        point1 = min(point1, len(sentences1))
        point2 = min(point2, len(sentences2))

        # Create offspring
        offspring1_parts = sentences1[:point1] + sentences2[point2:]
        offspring2_parts = sentences2[:point2] + sentences1[point1:]

        offspring1 = " ".join(offspring1_parts)
        offspring2 = " ".join(offspring2_parts)

        return offspring1, offspring2

    def _mutate_sync(self, content: str) -> str:
        """Synchronous mutation operation."""
        mutation_type = random.choice(["rephrase", "add", "remove", "modify"])

        if mutation_type == "rephrase":
            return self._rephrase_mutation(content)
        elif mutation_type == "add":
            return self._add_mutation(content)
        elif mutation_type == "remove":
            return self._remove_mutation(content)
        else:  # modify
            return self._modify_mutation(content)

    def _extract_sentences(self, content: str) -> list[str]:
        """Extract sentences or key points from content."""
        # Try to split by sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            # Try splitting by newlines or bullet points
            sentences = re.split(r'[\n•\-*]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _rephrase_mutation(self, content: str) -> str:
        """Rephrase part of the content."""
        sentences = self._extract_sentences(content)
        if not sentences:
            return content

        # Select random sentence to modify
        idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[idx]

        # Enhanced rephrasing with more variations
        variations = [
            f"Additionally, {sentence.lower()}",
            f"Furthermore, {sentence.lower()}",
            f"It's worth noting that {sentence.lower()}",
            f"Importantly, {sentence.lower()}",
            f"{sentence} This is crucial because it enables further innovation.",
            f"{sentence} This approach offers significant advantages.",
            f"Building on this, {sentence.lower()}",
            f"To elaborate, {sentence.lower()}",
            f"{sentence} This creates opportunities for optimization.",
            f"Specifically, {sentence.lower()}"
        ]

        sentences[idx] = random.choice(variations)
        return " ".join(sentences)

    def _add_mutation(self, content: str) -> str:
        """Add new element to content."""
        additions = [
            "Consider the scalability implications of this approach.",
            "This could be enhanced with machine learning techniques.",
            "User feedback would be essential for validation.",
            "Performance optimization should be a key consideration.",
            "Security aspects need careful attention.",
            "This aligns with current industry best practices.",
            "Real-time processing capabilities could add value.",
            "Integration with existing systems should be seamless.",
            "The solution should be designed for extensibility.",
            "Monitoring and analytics would provide valuable insights."
        ]

        return f"{content} {random.choice(additions)}"

    def _remove_mutation(self, content: str) -> str:
        """Remove element from content."""
        sentences = self._extract_sentences(content)
        if len(sentences) <= 1:
            return content

        # Remove random sentence
        idx = random.randint(0, len(sentences) - 1)
        sentences.pop(idx)
        return " ".join(sentences)

    def _modify_mutation(self, content: str) -> str:
        """Modify specific aspects of content."""
        modifications = [
            ("small", "large"),
            ("simple", "complex"),
            ("basic", "advanced"),
            ("traditional", "innovative"),
            ("sequential", "parallel"),
            ("manual", "automated"),
            ("static", "dynamic"),
            ("synchronous", "asynchronous"),
            ("centralized", "distributed"),
            ("monolithic", "modular")
        ]

        modified = content
        for old, new in modifications:
            if old in modified.lower():
                # Replace with proper case handling
                modified = re.sub(
                    rf'\b{old}\b',
                    new,
                    modified,
                    flags=re.IGNORECASE
                )
                break

        return modified

    async def adaptive_mutation_rate(
        self,
        generation: int,
        fitness_variance: float,
        convergence_rate: float
    ) -> float:
        """Calculate adaptive mutation rate based on population diversity."""
        base_rate = self.parameters.mutation_rate

        # Increase mutation if population is converging
        if convergence_rate > 0.8:
            # Population is converging, increase mutation
            adaptive_rate = min(base_rate * 2.0, 0.5)
        elif convergence_rate < 0.3:
            # Population is diverse, reduce mutation
            adaptive_rate = base_rate * 0.5
        else:
            # Normal diversity
            adaptive_rate = base_rate

        # Gradually increase mutation in later generations
        generation_factor = 1.0 + (generation * 0.05)
        adaptive_rate = min(adaptive_rate * generation_factor, 0.5)

        return adaptive_rate

    def calculate_population_diversity(
        self,
        fitness_scores: np.ndarray,
        embeddings: np.ndarray | None = None
    ) -> dict[str, float]:
        """Calculate diversity metrics for the population."""
        # Fitness diversity
        fitness_variance = np.var(fitness_scores)
        fitness_std = np.std(fitness_scores)

        # Convergence rate (how similar fitness scores are)
        if fitness_scores.max() > 0:
            convergence_rate = fitness_scores.mean() / fitness_scores.max()
        else:
            convergence_rate = 1.0

        metrics = {
            "fitness_variance": float(fitness_variance),
            "fitness_std": float(fitness_std),
            "convergence_rate": float(convergence_rate)
        }

        # Embedding diversity if available
        if embeddings is not None and len(embeddings) > 1:
            # Calculate pairwise distances
            distances = self.gpu_components["fitness_evaluator"].compute_pairwise_distances(embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = distances[np.triu_indices_from(distances, k=1)]

            metrics["embedding_diversity"] = float(upper_triangle.mean())
            metrics["embedding_diversity_std"] = float(upper_triangle.std())

        return metrics
