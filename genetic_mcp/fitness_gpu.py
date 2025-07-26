"""GPU-optimized fitness evaluation for ideas with async integration."""

import asyncio
import logging
from typing import Any

import numpy as np

from .gpu_accelerated import (
    GPUConfig,
    create_gpu_accelerated_components,
)
from .models import FitnessWeights, Idea

logger = logging.getLogger(__name__)


class GPUOptimizedFitnessEvaluator:
    """GPU-optimized fitness evaluator with async support."""

    def __init__(self, weights: FitnessWeights | None = None, gpu_config: GPUConfig | None = None):
        self.weights = weights or FitnessWeights()
        self.gpu_config = gpu_config or GPUConfig()

        # Initialize GPU components
        self.gpu_components = create_gpu_accelerated_components(self.gpu_config)
        self.batch_processor = self.gpu_components["batch_processor"]

        # Async lock for thread safety
        self._processing_lock = asyncio.Lock()

        logger.info(f"Initialized GPU-optimized fitness evaluator on {self.gpu_config.device}")

    async def evaluate_population_async(
        self,
        ideas: list[Idea],
        target_prompt: str,
        batch_size: int | None = None
    ) -> None:
        """Evaluate fitness for entire population asynchronously."""
        if not ideas:
            return

        batch_size = batch_size or self.gpu_config.batch_size

        async with self._processing_lock:
            # Process in batches if population is very large
            for i in range(0, len(ideas), batch_size):
                batch = ideas[i:i + batch_size]
                await self._process_batch(batch, target_prompt)

    async def _process_batch(self, ideas: list[Idea], target_prompt: str) -> None:
        """Process a batch of ideas."""
        # Convert ideas to dict format for batch processor
        idea_dicts = [
            {"id": idea.id, "content": idea.content}
            for idea in ideas
        ]

        # Get fitness weights as dict
        weights_dict = {
            "relevance": self.weights.relevance,
            "novelty": self.weights.novelty,
            "feasibility": self.weights.feasibility
        }

        # Process batch on GPU
        fitness_scores, score_components, embeddings = await self.batch_processor.process_population_batch(
            idea_dicts,
            target_prompt,
            weights_dict
        )

        # Update idea objects with results
        for idx, idea in enumerate(ideas):
            idea.fitness = float(fitness_scores[idx])
            idea.scores = {
                "relevance": float(score_components["relevance"][idx]),
                "novelty": float(score_components["novelty"][idx]),
                "feasibility": float(score_components["feasibility"][idx])
            }

    async def calculate_similarity_matrix(self, ideas: list[Idea]) -> np.ndarray:
        """Calculate similarity matrix between all ideas."""
        if not ideas:
            return np.array([])

        # Get embeddings from cache
        embeddings = []
        for idea in ideas:
            if idea.id in self.batch_processor.embedding_generator.embedding_cache:
                embedding = self.batch_processor.embedding_generator.embedding_cache[idea.id]
                embeddings.append(embedding.cpu().numpy())
            else:
                # Generate if not in cache
                emb_dict = await self.batch_processor.embedding_generator.generate_embeddings(
                    [idea.content], [idea.id]
                )
                embeddings.append(emb_dict[idea.id])

        embeddings = np.vstack(embeddings)

        # Compute similarity matrix on GPU
        distances = self.batch_processor.fitness_evaluator.compute_pairwise_distances(embeddings)

        # Convert distances to similarities
        max_distance = distances.max()
        if max_distance > 0:
            similarities = 1.0 - (distances / max_distance)
        else:
            similarities = np.ones_like(distances)

        return similarities

    async def find_diverse_subset(
        self,
        ideas: list[Idea],
        k: int,
        diversity_weight: float = 0.5
    ) -> list[Idea]:
        """Find k most diverse ideas with good fitness scores."""
        if len(ideas) <= k:
            return ideas

        # Calculate similarity matrix
        similarity_matrix = await self.calculate_similarity_matrix(ideas)

        # Greedy selection for diversity
        selected_indices = []
        remaining_indices = list(range(len(ideas)))

        # Start with highest fitness idea
        fitness_scores = np.array([idea.fitness for idea in ideas])
        first_idx = np.argmax(fitness_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Select remaining ideas
        while len(selected_indices) < k and remaining_indices:
            scores = []

            for idx in remaining_indices:
                # Calculate minimum similarity to selected ideas
                min_similarity = min(
                    similarity_matrix[idx, selected_idx]
                    for selected_idx in selected_indices
                )

                # Combine fitness and diversity
                diversity_score = 1.0 - min_similarity
                combined_score = (
                    (1 - diversity_weight) * fitness_scores[idx] +
                    diversity_weight * diversity_score
                )
                scores.append(combined_score)

            # Select idea with highest combined score
            best_idx = remaining_indices[np.argmax(scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [ideas[idx] for idx in selected_indices]

    def get_selection_probabilities_gpu(self, ideas: list[Idea]) -> np.ndarray:
        """Get selection probabilities using GPU-accelerated softmax."""
        if not ideas:
            return np.array([])

        fitness_scores = np.array([idea.fitness for idea in ideas])

        if self.batch_processor.fitness_evaluator.use_gpu:
            import torch
            # Use temperature-scaled softmax for better diversity
            temperature = 2.0  # Higher temperature = more uniform distribution
            scores_tensor = torch.from_numpy(fitness_scores).to(self.gpu_config.device)
            probabilities = torch.softmax(scores_tensor / temperature, dim=0)
            return probabilities.cpu().numpy()
        else:
            # CPU fallback
            # Avoid numerical overflow
            scores_shifted = fitness_scores - fitness_scores.max()
            exp_scores = np.exp(scores_shifted / 2.0)
            return exp_scores / exp_scores.sum()

    async def parallel_tournament_selection(
        self,
        ideas: list[Idea],
        num_selections: int,
        tournament_size: int = 3
    ) -> list[Idea]:
        """Perform tournament selection in parallel on GPU."""
        if not ideas:
            return []

        fitness_scores = np.array([idea.fitness for idea in ideas])

        # Use GPU-accelerated tournament selection
        winner_indices = self.batch_processor.genetic_ops.tournament_selection_batch(
            fitness_scores,
            num_selections,
            tournament_size
        )

        return [ideas[idx] for idx in winner_indices]

    def get_memory_stats(self) -> dict[str, Any]:
        """Get GPU memory statistics."""
        return self.batch_processor.get_memory_stats()

    def clear_caches(self) -> None:
        """Clear GPU memory caches."""
        self.batch_processor.clear_caches()


class AsyncFitnessEvaluatorWrapper:
    """Wrapper to make GPU fitness evaluator work with existing async code."""

    def __init__(self, gpu_evaluator: GPUOptimizedFitnessEvaluator):
        self.gpu_evaluator = gpu_evaluator
        self.embeddings_cache = {}  # Compatibility layer

    async def calculate_fitness(
        self,
        idea: Idea,
        all_ideas: list[Idea],
        target_embedding: list[float]
    ) -> float:
        """Calculate fitness for a single idea (compatibility method)."""
        # For single idea evaluation, we still process as batch for efficiency
        target_prompt = "Target"  # This should be passed properly
        await self.gpu_evaluator.evaluate_population_async([idea], target_prompt)
        return idea.fitness

    async def evaluate_population(
        self,
        ideas: list[Idea],
        target_prompt: str
    ) -> None:
        """Evaluate entire population."""
        await self.gpu_evaluator.evaluate_population_async(ideas, target_prompt)

    def add_embedding(self, idea_id: str, embedding: list[float]) -> None:
        """Add embedding to cache (compatibility method)."""
        # This is handled internally by GPU batch processor
        pass

    def get_selection_probabilities(self, ideas: list[Idea]) -> list[float]:
        """Get selection probabilities."""
        return self.gpu_evaluator.get_selection_probabilities_gpu(ideas).tolist()

    async def tournament_select(
        self,
        ideas: list[Idea],
        tournament_size: int = 3
    ) -> Idea:
        """Select single idea using tournament selection."""
        selected = await self.gpu_evaluator.parallel_tournament_selection(
            ideas, 1, tournament_size
        )
        return selected[0] if selected else None
