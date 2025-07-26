"""Integration module for GPU-accelerated genetic algorithm with async MCP server."""

import asyncio
import logging
from datetime import datetime
from typing import Any

import numpy as np

from .fitness_gpu import GPUOptimizedFitnessEvaluator
from .genetic_algorithm_gpu import GPUOptimizedGeneticAlgorithm
from .gpu_accelerated import GPUConfig, create_gpu_accelerated_components
from .models import (
    EvolutionMode,
    FitnessWeights,
    GenerationProgress,
    GeneticParameters,
    Idea,
    Session,
)

logger = logging.getLogger(__name__)


class GPUAcceleratedSessionManager:
    """Session manager with GPU acceleration for genetic operations."""

    def __init__(
        self,
        gpu_config: GPUConfig | None = None,
        enable_gpu: bool = True
    ):
        self.gpu_config = gpu_config or GPUConfig()
        self.enable_gpu = enable_gpu and self.gpu_config.device != "cpu"

        # Initialize GPU components if enabled
        if self.enable_gpu:
            try:
                self.gpu_components = create_gpu_accelerated_components(self.gpu_config)
                self.fitness_evaluator = GPUOptimizedFitnessEvaluator(gpu_config=self.gpu_config)
                self.genetic_algorithm = GPUOptimizedGeneticAlgorithm(gpu_config=self.gpu_config)
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU components: {e}")
                self.enable_gpu = False

        if not self.enable_gpu:
            logger.info("Running in CPU-only mode")

    async def process_generation_gpu(
        self,
        session: Session,
        current_ideas: list[Idea],
        target_prompt: str
    ) -> tuple[list[Idea], GenerationProgress]:
        """Process a generation using GPU acceleration."""
        generation_start = datetime.utcnow()

        # Update fitness evaluator weights if needed
        if self.enable_gpu:
            self.fitness_evaluator.weights = session.fitness_weights
            self.genetic_algorithm.parameters = session.parameters

        # Evaluate fitness for current population
        if self.enable_gpu:
            await self.fitness_evaluator.evaluate_population_async(
                current_ideas,
                target_prompt
            )
        else:
            # Fallback to CPU implementation
            await self._evaluate_population_cpu(current_ideas, target_prompt, session.fitness_weights)

        # Get fitness scores
        fitness_scores = np.array([idea.fitness for idea in current_ideas])

        # Calculate diversity metrics
        diversity_metrics = {}
        if self.enable_gpu:
            diversity_metrics = self.genetic_algorithm.calculate_population_diversity(
                fitness_scores
            )

        # Create progress update
        progress = GenerationProgress(
            session_id=session.id,
            current_generation=session.current_generation,
            total_generations=session.parameters.generations,
            ideas_generated=len(session.ideas),
            active_workers=len(session.get_active_workers()),
            best_fitness=float(fitness_scores.max()) if len(fitness_scores) > 0 else 0.0,
            status="evolving",
            message=f"Generation {session.current_generation}: diversity={diversity_metrics.get('convergence_rate', 0):.2f}"
        )

        # If not last generation, create next generation
        if session.current_generation < session.parameters.generations:
            if self.enable_gpu:
                # GPU-accelerated evolution
                new_population = await self.genetic_algorithm.create_next_generation_gpu(
                    current_ideas,
                    fitness_scores,
                    session.current_generation + 1,
                    target_prompt
                )
            else:
                # Fallback to CPU implementation
                new_population = await self._create_next_generation_cpu(
                    session,
                    current_ideas,
                    fitness_scores,
                    session.current_generation + 1
                )

            # Update session
            session.ideas.extend(new_population)
            session.current_generation += 1

            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            logger.info(f"Generation {session.current_generation} completed in {generation_time:.2f}s")

            return new_population, progress

        return [], progress

    async def find_optimal_population_subset(
        self,
        session: Session,
        k: int,
        diversity_weight: float = 0.5
    ) -> list[Idea]:
        """Find optimal subset of ideas balancing fitness and diversity."""
        if not session.ideas:
            return []

        if self.enable_gpu:
            # Use GPU-accelerated diversity selection
            return await self.fitness_evaluator.find_diverse_subset(
                session.ideas,
                k,
                diversity_weight
            )
        else:
            # Simple CPU fallback - just return top-k by fitness
            sorted_ideas = sorted(session.ideas, key=lambda x: x.fitness, reverse=True)
            return sorted_ideas[:k]

    def get_gpu_memory_stats(self) -> dict[str, Any]:
        """Get current GPU memory statistics."""
        if not self.enable_gpu:
            return {"gpu_enabled": False}

        stats = self.fitness_evaluator.get_memory_stats()
        stats["gpu_enabled"] = True
        stats["device"] = self.gpu_config.device

        return stats

    def clear_gpu_caches(self) -> None:
        """Clear GPU memory caches."""
        if self.enable_gpu:
            self.fitness_evaluator.clear_caches()
            logger.info("GPU caches cleared")

    async def warm_up_gpu(self, sample_size: int = 10) -> None:
        """Warm up GPU with sample computations."""
        if not self.enable_gpu:
            return

        logger.info("Warming up GPU...")

        # Create sample data
        sample_ideas = [
            Idea(
                id=f"warmup_{i}",
                content=f"Sample idea {i} for GPU warmup. This is a test content to initialize GPU operations.",
                generation=0
            )
            for i in range(sample_size)
        ]

        # Run sample evaluation
        await self.fitness_evaluator.evaluate_population_async(
            sample_ideas,
            "Sample target prompt for warmup"
        )

        # Clear warmup data
        self.fitness_evaluator.clear_caches()

        logger.info("GPU warmup completed")

    async def _evaluate_population_cpu(
        self,
        ideas: list[Idea],
        target_prompt: str,
        weights: FitnessWeights
    ) -> None:
        """CPU fallback for fitness evaluation."""
        # This would use the original CPU implementation
        # For now, just assign random fitness scores
        for idea in ideas:
            idea.fitness = np.random.random()
            idea.scores = {
                "relevance": np.random.random(),
                "novelty": np.random.random(),
                "feasibility": np.random.random()
            }

    async def _create_next_generation_cpu(
        self,
        session: Session,
        population: list[Idea],
        fitness_scores: np.ndarray,
        generation: int
    ) -> list[Idea]:
        """CPU fallback for creating next generation."""
        # This would use the original CPU implementation
        # For now, just return a copy of the best ideas
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = []

        for i in range(min(len(population), session.parameters.population_size)):
            idx = sorted_indices[i % len(sorted_indices)]
            old_idea = population[idx]
            new_idea = Idea(
                id=f"gen{generation}_cpu_{i}",
                content=old_idea.content + f" [Gen {generation}]",
                generation=generation,
                parent_ids=[old_idea.id]
            )
            new_population.append(new_idea)

        return new_population


class GPUBatchIdeaProcessor:
    """Batch processor for handling multiple idea generation sessions efficiently."""

    def __init__(self, gpu_config: GPUConfig | None = None):
        self.gpu_config = gpu_config or GPUConfig()
        self.batch_processor = create_gpu_accelerated_components(self.gpu_config)["batch_processor"]
        self._processing_queue = asyncio.Queue()
        self._batch_task = None

    async def start(self) -> None:
        """Start the batch processor."""
        self._batch_task = asyncio.create_task(self._batch_processing_loop())
        logger.info("GPU batch processor started")

    async def stop(self) -> None:
        """Stop the batch processor."""
        if self._batch_task:
            self._batch_task.cancel()
            await asyncio.gather(self._batch_task, return_exceptions=True)
        logger.info("GPU batch processor stopped")

    async def add_to_batch(
        self,
        session_id: str,
        ideas: list[Idea],
        target_prompt: str,
        weights: dict[str, float]
    ) -> asyncio.Future:
        """Add ideas to processing batch."""
        future = asyncio.Future()
        await self._processing_queue.put((session_id, ideas, target_prompt, weights, future))
        return future

    async def _batch_processing_loop(self) -> None:
        """Process batches of ideas efficiently."""
        while True:
            try:
                # Collect items for batch processing
                batch = []
                deadline = asyncio.get_event_loop().time() + 0.1  # 100ms batching window

                while len(batch) < self.gpu_config.batch_size:
                    timeout = max(0, deadline - asyncio.get_event_loop().time())
                    if timeout <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(
                            self._processing_queue.get(),
                            timeout=timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    await self._process_batch(batch)
                else:
                    await asyncio.sleep(0.01)  # Brief pause if no items

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def _process_batch(self, batch: list[tuple]) -> None:
        """Process a batch of idea evaluation requests."""
        # Combine all ideas from different sessions
        all_ideas = []
        all_prompts = []
        session_mapping = []

        for session_id, ideas, target_prompt, weights, future in batch:
            start_idx = len(all_ideas)
            all_ideas.extend(ideas)
            all_prompts.extend([target_prompt] * len(ideas))
            session_mapping.append((session_id, start_idx, len(ideas), weights, future))

        try:
            # Process all ideas in one GPU batch
            # This is more efficient than processing each session separately
            idea_dicts = [{"id": idea.id, "content": idea.content} for idea in all_ideas]

            # Use the first session's weights (or implement weighted average)
            combined_weights = batch[0][3]

            # Process on GPU
            fitness_scores, components, embeddings = await self.batch_processor.process_population_batch(
                idea_dicts,
                all_prompts[0],  # Assuming same prompt for now
                combined_weights
            )

            # Distribute results back to sessions
            for _, start_idx, count, _, future in session_mapping:
                end_idx = start_idx + count
                session_ideas = all_ideas[start_idx:end_idx]
                session_fitness = fitness_scores[start_idx:end_idx]
                session_components = {
                    k: v[start_idx:end_idx] for k, v in components.items()
                }

                # Update idea objects
                for i, idea in enumerate(session_ideas):
                    idea.fitness = float(session_fitness[i])
                    idea.scores = {
                        k: float(v[i]) for k, v in session_components.items()
                    }

                # Complete the future
                future.set_result({
                    "fitness_scores": session_fitness,
                    "components": session_components,
                    "ideas": session_ideas
                })

        except Exception as e:
            # Fail all futures in the batch
            for _, _, _, _, future in session_mapping:
                if not future.done():
                    future.set_exception(e)


# Example usage function
async def example_gpu_accelerated_evolution():
    """Example of using GPU-accelerated genetic algorithm."""
    # Configure GPU
    gpu_config = GPUConfig(
        device="cuda",  # or "cpu" for testing
        batch_size=64,
        use_mixed_precision=True,
        memory_fraction=0.8
    )

    # Create session manager with GPU acceleration
    gpu_session_manager = GPUAcceleratedSessionManager(gpu_config=gpu_config)

    # Warm up GPU
    await gpu_session_manager.warm_up_gpu()

    # Create sample session
    session = Session(
        id="test_session",
        client_id="test_client",
        prompt="Generate innovative ideas for sustainable energy",
        mode=EvolutionMode.ITERATIVE,
        parameters=GeneticParameters(
            population_size=50,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_count=5
        ),
        fitness_weights=FitnessWeights(
            relevance=0.4,
            novelty=0.3,
            feasibility=0.3
        )
    )

    # Create initial population
    initial_ideas = [
        Idea(
            id=f"initial_{i}",
            content=f"Idea {i}: Solar panels with {i}% efficiency improvement",
            generation=0
        )
        for i in range(50)
    ]
    session.ideas = initial_ideas

    # Run evolution
    for gen in range(session.parameters.generations):
        current_population = session.ideas[-session.parameters.population_size:]
        new_population, progress = await gpu_session_manager.process_generation_gpu(
            session,
            current_population,
            session.prompt
        )

        logger.info(f"Generation {gen}: Best fitness = {progress.best_fitness:.3f}")

    # Get diverse top ideas
    top_ideas = await gpu_session_manager.find_optimal_population_subset(
        session,
        k=10,
        diversity_weight=0.5
    )

    # Print GPU memory stats
    stats = gpu_session_manager.get_gpu_memory_stats()
    logger.info(f"GPU Memory Stats: {stats}")

    # Clean up
    gpu_session_manager.clear_gpu_caches()

    return top_ideas


if __name__ == "__main__":
    # Run example
    asyncio.run(example_gpu_accelerated_evolution())
