"""GPU-optimized batch evolution for processing multiple populations simultaneously.

This module enables efficient parallel evolution of multiple independent populations
or experiments, maximizing GPU utilization through batched operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .gpu_accelerated import GPUConfig, create_gpu_accelerated_components
from .gpu_diversity_metrics import GPUDiversityMetrics
from .gpu_selection_optimized import GPUOptimizedSelection
from .models import GeneticParameters, Idea

logger = logging.getLogger(__name__)


@dataclass
class BatchEvolutionConfig:
    """Configuration for batch evolution processing."""
    n_experiments: int = 10
    max_batch_size: int = 100  # Maximum ideas to process in single GPU batch
    async_workers: int = 4
    memory_limit_gb: float = 10.0
    checkpoint_interval: int = 10  # Generations between checkpoints


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in the batch."""
    experiment_id: str
    population_size: int
    target_prompt: str
    parameters: GeneticParameters
    metadata: dict[str, Any] = None


class GPUBatchEvolution:
    """Manages batch evolution of multiple populations on GPU."""

    def __init__(
        self,
        batch_config: BatchEvolutionConfig = None,
        gpu_config: GPUConfig = None
    ):
        self.batch_config = batch_config or BatchEvolutionConfig()
        self.gpu_config = gpu_config or GPUConfig()

        # Initialize GPU components
        self.gpu_components = create_gpu_accelerated_components(self.gpu_config)
        self.selection_optimizer = GPUOptimizedSelection(self.gpu_config)
        self.diversity_metrics = GPUDiversityMetrics(self.gpu_config)

        # Batch processing state
        self.active_experiments: dict[str, ExperimentState] = {}
        self.processing_queue = asyncio.Queue()
        self.results_cache: dict[str, Any] = {}

        # Resource management
        self.memory_monitor = GPUMemoryMonitor(self.gpu_config)
        self._processing_lock = asyncio.Lock()

        logger.info(f"Initialized batch evolution processor on {self.gpu_config.device}")

    async def run_batch_experiments(
        self,
        experiment_configs: list[ExperimentConfig],
        generations: int,
        callbacks: dict[str, Any] | None = None
    ) -> dict[str, list[Idea]]:
        """Run multiple experiments in parallel with GPU batching."""
        # Initialize experiments
        for config in experiment_configs:
            self.active_experiments[config.experiment_id] = ExperimentState(
                config=config,
                generation=0,
                population=[],
                fitness_history=[],
                diversity_history=[]
            )

        # Process generations
        for gen in range(generations):
            logger.info(f"Batch processing generation {gen + 1}/{generations}")

            # Check memory before processing
            if not await self.memory_monitor.check_memory_available():
                await self._cleanup_memory()

            # Process all experiments for this generation
            await self._process_generation_batch(gen)

            # Checkpoint if needed
            if gen % self.batch_config.checkpoint_interval == 0:
                await self._checkpoint_experiments()

            # Callbacks
            if callbacks and 'on_batch_generation' in callbacks:
                await callbacks['on_batch_generation'](gen, self.active_experiments)

        # Collect final results
        results = {}
        for exp_id, state in self.active_experiments.items():
            results[exp_id] = state.population

        return results

    async def _process_generation_batch(self, generation: int) -> None:
        """Process one generation for all experiments using GPU batching."""
        # Collect all individuals across experiments
        all_ideas = []
        experiment_mapping = []  # Track which idea belongs to which experiment

        for exp_id, state in self.active_experiments.items():
            if generation == 0:
                # Initialize population
                population = await self._initialize_population(state.config)
                state.population = population

            for idea in state.population:
                all_ideas.append(idea)
                experiment_mapping.append(exp_id)

        # Batch fitness evaluation
        await self._batch_evaluate_fitness(all_ideas, experiment_mapping)

        # Calculate diversity metrics in batch
        diversity_results = await self._batch_calculate_diversity(all_ideas, experiment_mapping)

        # Update experiment states
        idea_idx = 0
        for exp_id, state in self.active_experiments.items():
            pop_size = len(state.population)

            # Extract fitness and diversity for this experiment
            exp_fitness = [all_ideas[idea_idx + i].fitness for i in range(pop_size)]
            state.fitness_history.append(exp_fitness)
            state.diversity_history.append(diversity_results[exp_id])

            idea_idx += pop_size

        # Batch selection and reproduction
        await self._batch_create_next_generation(generation)

    async def _batch_evaluate_fitness(
        self,
        all_ideas: list[Idea],
        experiment_mapping: list[str]
    ) -> None:
        """Evaluate fitness for all ideas across experiments in batches."""
        # Group by target prompt for efficient batching
        prompt_groups: dict[str, list[tuple[int, Idea]]] = {}

        for idx, (idea, exp_id) in enumerate(zip(all_ideas, experiment_mapping, strict=False)):
            target_prompt = self.active_experiments[exp_id].config.target_prompt
            if target_prompt not in prompt_groups:
                prompt_groups[target_prompt] = []
            prompt_groups[target_prompt].append((idx, idea))

        # Process each prompt group
        for target_prompt, idea_group in prompt_groups.items():
            indices, ideas = zip(*idea_group, strict=False)

            # Batch process on GPU
            for i in range(0, len(ideas), self.batch_config.max_batch_size):
                batch_ideas = list(ideas[i:i+self.batch_config.max_batch_size])
                batch_indices = list(indices[i:i+self.batch_config.max_batch_size])

                # Get embeddings and fitness in batch
                texts = [idea.content for idea in batch_ideas]
                ids = [idea.id for idea in batch_ideas]

                # Process through GPU pipeline
                embeddings_dict = await self.gpu_components["embedding_generator"].generate_embeddings(
                    texts + [target_prompt], ids + ['target']
                )

                idea_embeddings = np.vstack([embeddings_dict[id_] for id_ in ids])
                target_embedding = embeddings_dict['target']

                # Calculate fitness components
                fitness_scores, components = self.gpu_components["fitness_evaluator"].batch_evaluate_fitness(
                    idea_embeddings,
                    target_embedding,
                    {'relevance': 0.4, 'novelty': 0.3, 'feasibility': 0.3}
                )

                # Update ideas with results
                for j, (idea_idx, _) in enumerate(zip(batch_indices, batch_ideas, strict=False)):
                    all_ideas[idea_idx].fitness = float(fitness_scores[j])
                    all_ideas[idea_idx].scores = {
                        'relevance': float(components['relevance'][j]),
                        'novelty': float(components['novelty'][j]),
                        'feasibility': float(components['feasibility'][j])
                    }

    async def _batch_calculate_diversity(
        self,
        all_ideas: list[Idea],
        experiment_mapping: list[str]
    ) -> dict[str, dict[str, float]]:
        """Calculate diversity metrics for all experiments in batch."""
        diversity_results = {}

        # Group ideas by experiment
        experiment_ideas: dict[str, list[Idea]] = {}
        for idea, exp_id in zip(all_ideas, experiment_mapping, strict=False):
            if exp_id not in experiment_ideas:
                experiment_ideas[exp_id] = []
            experiment_ideas[exp_id].append(idea)

        # Batch calculate diversity
        if self.gpu_config.device != "cpu" and TORCH_AVAILABLE:
            # Process all experiments in single GPU call
            all_embeddings = []
            all_fitness = []
            exp_sizes = []

            for exp_id, ideas in experiment_ideas.items():
                # Get embeddings from cache
                embeddings = []
                fitness = []
                for idea in ideas:
                    if idea.id in self.gpu_components["embedding_generator"].embedding_cache:
                        emb = self.gpu_components["embedding_generator"].embedding_cache[idea.id]
                        embeddings.append(emb.cpu().numpy())
                        fitness.append(idea.fitness)

                if embeddings:
                    all_embeddings.append(np.vstack(embeddings))
                    all_fitness.append(np.array(fitness))
                    exp_sizes.append((exp_id, len(embeddings)))

            # Batch process diversity metrics
            if all_embeddings:
                batch_diversity = self.diversity_metrics.batch_diversity_calculation_gpu(
                    all_embeddings, all_fitness
                )

                for (exp_id, _), metrics in zip(exp_sizes, batch_diversity, strict=False):
                    diversity_results[exp_id] = metrics
        else:
            # CPU fallback
            for exp_id, ideas in experiment_ideas.items():
                embeddings = []
                fitness = []
                for idea in ideas:
                    # Simple embedding approximation for CPU
                    embeddings.append(np.random.randn(768))  # Placeholder
                    fitness.append(idea.fitness)

                embeddings = np.vstack(embeddings)
                fitness = np.array(fitness)

                diversity_results[exp_id] = {
                    'embedding_diversity': float(np.std(embeddings)),
                    'fitness_variance': float(np.var(fitness))
                }

        return diversity_results

    async def _batch_create_next_generation(self, generation: int) -> None:
        """Create next generation for all experiments using batched operations."""
        # Collect selection tasks
        selection_tasks = []

        for exp_id, state in self.active_experiments.items():
            task = self._create_next_generation_for_experiment(
                exp_id, state, generation
            )
            selection_tasks.append(task)

        # Process in parallel with limited concurrency
        sem = asyncio.Semaphore(self.batch_config.async_workers)

        async def limited_task(task):
            async with sem:
                return await task

        new_populations = await asyncio.gather(
            *[limited_task(task) for task in selection_tasks]
        )

        # Update populations
        for (exp_id, _), new_pop in zip(self.active_experiments.items(), new_populations, strict=False):
            self.active_experiments[exp_id].population = new_pop

    async def _create_next_generation_for_experiment(
        self,
        exp_id: str,
        state: 'ExperimentState',
        generation: int
    ) -> list[Idea]:
        """Create next generation for a single experiment."""
        population = state.population
        parameters = state.config.parameters

        # Get fitness scores
        fitness_scores = np.array([idea.fitness for idea in population])

        # GPU-accelerated selection
        if parameters.selection_method == "tournament":
            parent_indices = self.selection_optimizer.batch_tournament_selection_advanced_gpu(
                fitness_scores,
                parameters.population_size,
                tournament_size=3
            )
        else:
            # Default to tournament
            parent_indices = self.selection_optimizer.batch_tournament_selection_advanced_gpu(
                fitness_scores,
                parameters.population_size
            )

        # Create offspring (simplified for batch processing)
        new_population = []
        for i in range(0, len(parent_indices), 2):
            if i + 1 < len(parent_indices):
                parent1 = population[parent_indices[i]]
                parent2 = population[parent_indices[i + 1]]

                # Simple crossover
                offspring_content = self._simple_crossover_batch(
                    parent1.content, parent2.content
                )

                # Simple mutation
                if np.random.random() < parameters.mutation_rate:
                    offspring_content = self._simple_mutation_batch(offspring_content)

                offspring = Idea(
                    id=f"{exp_id}_gen{generation}_off{i}",
                    content=offspring_content,
                    generation=generation,
                    parent_ids=[parent1.id, parent2.id]
                )
                new_population.append(offspring)

        # Ensure population size
        while len(new_population) < parameters.population_size:
            # Clone random individual
            template = np.random.choice(population)
            clone = Idea(
                id=f"{exp_id}_gen{generation}_clone{len(new_population)}",
                content=template.content,
                generation=generation,
                parent_ids=[template.id]
            )
            new_population.append(clone)

        return new_population[:parameters.population_size]

    def _simple_crossover_batch(self, content1: str, content2: str) -> str:
        """Simple crossover for batch processing."""
        words1 = content1.split()
        words2 = content2.split()

        if len(words1) > 4 and len(words2) > 4:
            mid1 = len(words1) // 2
            mid2 = len(words2) // 2
            offspring = words1[:mid1] + words2[mid2:]
            return ' '.join(offspring)

        return content1 if np.random.random() < 0.5 else content2

    def _simple_mutation_batch(self, content: str) -> str:
        """Simple mutation for batch processing."""
        mutations = [
            " with enhanced performance",
            " using advanced techniques",
            " for optimal results",
            " with improved efficiency",
            " featuring novel approaches"
        ]
        return content + np.random.choice(mutations)

    async def _initialize_population(self, config: ExperimentConfig) -> list[Idea]:
        """Initialize population for an experiment."""
        population = []
        base_ideas = [
            f"Implement {config.target_prompt} using machine learning",
            f"Create a system for {config.target_prompt} with automation",
            f"Design an architecture for {config.target_prompt}",
            f"Develop {config.target_prompt} with scalability in mind",
            f"Build {config.target_prompt} using modern technologies"
        ]

        for i in range(config.population_size):
            idea = Idea(
                id=f"{config.experiment_id}_init_{i}",
                content=np.random.choice(base_ideas) + f" (variant {i})",
                generation=0
            )
            population.append(idea)

        return population

    async def _cleanup_memory(self) -> None:
        """Clean up GPU memory when needed."""
        logger.info("Cleaning up GPU memory")

        # Clear caches
        self.gpu_components["batch_processor"].clear_caches()

        # Force garbage collection if using PyTorch
        if TORCH_AVAILABLE and self.gpu_config.device == "cuda":
            torch.cuda.empty_cache()

        # Clear old results
        if len(self.results_cache) > 100:
            # Keep only recent results
            sorted_keys = sorted(self.results_cache.keys())
            for key in sorted_keys[:50]:
                del self.results_cache[key]

    async def _checkpoint_experiments(self) -> None:
        """Save checkpoint of all experiments."""
        checkpoint_data = {}

        for exp_id, state in self.active_experiments.items():
            checkpoint_data[exp_id] = {
                'generation': state.generation,
                'best_fitness': max([idea.fitness for idea in state.population]),
                'avg_fitness': np.mean([idea.fitness for idea in state.population]),
                'population_size': len(state.population)
            }

        logger.info(f"Checkpoint saved for {len(checkpoint_data)} experiments")
        self.results_cache[f'checkpoint_gen_{state.generation}'] = checkpoint_data

    def get_batch_summary(self) -> dict[str, Any]:
        """Get summary of batch evolution progress."""
        summary = {
            'n_experiments': len(self.active_experiments),
            'device': self.gpu_config.device,
            'experiments': {}
        }

        for exp_id, state in self.active_experiments.items():
            if state.fitness_history:
                summary['experiments'][exp_id] = {
                    'generation': state.generation,
                    'best_fitness': max(state.fitness_history[-1]) if state.fitness_history[-1] else 0,
                    'avg_fitness': np.mean(state.fitness_history[-1]) if state.fitness_history[-1] else 0,
                    'improvement': (max(state.fitness_history[-1]) - max(state.fitness_history[0]))
                                  if len(state.fitness_history) > 1 else 0
                }

        return summary


@dataclass
class ExperimentState:
    """State tracking for a single experiment."""
    config: ExperimentConfig
    generation: int
    population: list[Idea]
    fitness_history: list[list[float]]
    diversity_history: list[dict[str, float]]


class GPUMemoryMonitor:
    """Monitor GPU memory usage for batch processing."""

    def __init__(self, gpu_config: GPUConfig):
        self.gpu_config = gpu_config
        self.use_gpu = gpu_config.device != "cpu" and TORCH_AVAILABLE

    async def check_memory_available(self, required_gb: float = 1.0) -> bool:
        """Check if sufficient GPU memory is available."""
        if not self.use_gpu:
            return True

        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_gb = free_memory / (1024 ** 3)
            return free_gb >= required_gb
        except Exception:
            return True  # Assume available if can't check

    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        if not self.use_gpu:
            return {'allocated': 0, 'cached': 0, 'free': 0}

        return {
            'allocated': torch.cuda.memory_allocated() / (1024 ** 3),
            'cached': torch.cuda.memory_reserved() / (1024 ** 3),
            'free': (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated()) / (1024 ** 3)
        }
