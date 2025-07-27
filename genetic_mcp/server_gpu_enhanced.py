"""GPU-enhanced MCP server with advanced genetic algorithm features.

This module extends the base server with GPU acceleration and advanced
selection strategies for high-performance evolution.
"""

import logging
import os
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .genetic_algorithm_gpu_enhanced import AdvancedGeneticParameters
from .gpu_accelerated import GPUConfig
from .gpu_batch_evolution import BatchEvolutionConfig, ExperimentConfig
from .llm_client import MultiModelClient
from .models import (
    EvolutionMode,
    FitnessWeights,
    GenerationRequest,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class CreateAdvancedSessionSchema(BaseModel):
    """Schema for create_advanced_session tool with GPU options."""
    prompt: str = Field(description="The prompt to generate ideas for")
    mode: str = Field(
        default="single_pass",
        description="Evolution mode: 'single_pass' or 'iterative'"
    )
    population_size: int = Field(default=50, ge=10, description="Number of ideas per generation")
    top_k: int = Field(default=5, ge=1, description="Number of top ideas to return")
    generations: int = Field(default=20, ge=1, description="Number of generations")

    # Advanced selection options
    selection_method: str = Field(
        default="tournament",
        description="Selection method: tournament, boltzmann, sus, rank, diversity"
    )
    use_fitness_sharing: bool = Field(default=False, description="Enable fitness sharing for diversity")
    use_crowding: bool = Field(default=False, description="Enable crowding distance for multi-objective")
    use_pareto_ranking: bool = Field(default=False, description="Enable Pareto ranking")

    # GPU options
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration if available")
    gpu_batch_size: int = Field(default=64, description="Batch size for GPU processing")

    # Multi-population options
    n_subpopulations: int = Field(default=1, description="Number of subpopulations")
    migration_rate: float = Field(default=0.1, description="Migration rate between subpopulations")

    # Adaptive parameters
    adaptive_mutation: bool = Field(default=True, description="Enable adaptive mutation rate")
    adaptive_crossover: bool = Field(default=True, description="Enable adaptive crossover rate")

    fitness_weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for fitness calculation"
    )
    models: list[str] | None = Field(default=None, description="List of LLM models to use")


class BatchExperimentSchema(BaseModel):
    """Schema for batch experiment processing."""
    experiments: list[dict[str, Any]] = Field(
        description="List of experiment configurations"
    )
    generations: int = Field(default=20, description="Number of generations per experiment")
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    max_batch_size: int = Field(default=200, description="Maximum batch size for GPU")
    checkpoint_interval: int = Field(default=10, description="Generations between checkpoints")


class GPUEnhancedSessionManager(SessionManager):
    """Extended session manager with GPU optimization support."""

    def __init__(self, llm_client: MultiModelClient):
        super().__init__(llm_client)
        self.gpu_config = self._initialize_gpu_config()
        self.batch_processor = None

    def _initialize_gpu_config(self) -> GPUConfig:
        """Initialize GPU configuration based on environment."""
        device = "cuda" if os.getenv("GENETIC_MCP_GPU", "true").lower() == "true" else "cpu"

        # Check actual GPU availability
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("GPU requested but not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, GPU features disabled")
            device = "cpu"

        return GPUConfig(
            device=device,
            batch_size=int(os.getenv("GPU_BATCH_SIZE", "64")),
            use_mixed_precision=os.getenv("GPU_MIXED_PRECISION", "true").lower() == "true",
            memory_fraction=float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
        )

    async def create_advanced_session(
        self,
        client_id: str,
        request: GenerationRequest,
        advanced_params: AdvancedGeneticParameters
    ) -> Any:
        """Create session with advanced genetic algorithm parameters."""
        # Create base session
        session = await self.create_session(client_id, request)

        # Store advanced parameters
        session.metadata = session.metadata or {}
        session.metadata['advanced_params'] = advanced_params
        session.metadata['gpu_enabled'] = self.gpu_config.device != "cpu"

        return session

    async def run_batch_experiments(
        self,
        experiments: list[ExperimentConfig],
        generations: int,
        batch_config: BatchEvolutionConfig
    ) -> dict[str, Any]:
        """Run multiple experiments in batch with GPU acceleration."""
        if self.batch_processor is None:
            from .gpu_batch_evolution import GPUBatchEvolution
            self.batch_processor = GPUBatchEvolution(batch_config, self.gpu_config)

        # Run experiments
        results = await self.batch_processor.run_batch_experiments(
            experiments, generations
        )

        # Get summary
        summary = self.batch_processor.get_batch_summary()

        return {
            "results": results,
            "summary": summary
        }


# Initialize enhanced MCP server
mcp_gpu = FastMCP("genetic-mcp-gpu")


@mcp_gpu.tool()
async def create_advanced_session(
    prompt: str,
    mode: str = "iterative",
    population_size: int = 50,
    top_k: int = 5,
    generations: int = 20,
    selection_method: str = "tournament",
    use_fitness_sharing: bool = False,
    use_crowding: bool = False,
    use_pareto_ranking: bool = False,
    use_gpu: bool = True,
    gpu_batch_size: int = 64,
    n_subpopulations: int = 1,
    migration_rate: float = 0.1,
    adaptive_mutation: bool = True,
    adaptive_crossover: bool = True,
    fitness_weights: dict[str, float] | None = None,
    models: list[str] | None = None
) -> dict[str, Any]:
    """Create an advanced GA session with GPU optimization and advanced features.

    This tool provides access to GPU-accelerated genetic algorithms with:
    - Advanced selection strategies (Boltzmann, SUS, rank-based)
    - Fitness sharing and crowding distance for diversity
    - Multi-population evolution with migration
    - Adaptive parameter control
    - Batch processing on GPU for efficiency

    Returns:
        Session information with advanced configuration details
    """
    try:
        # Create advanced parameters
        advanced_params = AdvancedGeneticParameters(
            population_size=population_size,
            generations=generations,
            selection_method=selection_method,
            use_fitness_sharing=use_fitness_sharing,
            use_crowding=use_crowding,
            use_pareto_ranking=use_pareto_ranking,
            n_subpopulations=n_subpopulations,
            migration_rate=migration_rate,
            adaptive_mutation=adaptive_mutation,
            adaptive_crossover=adaptive_crossover
        )

        # Parse fitness weights
        weights = None
        if fitness_weights:
            weights = FitnessWeights(**fitness_weights)
            advanced_params.fitness_weights = weights

        # Create base request
        request = GenerationRequest(
            prompt=prompt,
            mode=EvolutionMode(mode),
            population_size=population_size,
            top_k=top_k,
            generations=generations,
            fitness_weights=weights,
            models=models
        )

        # Initialize session manager if needed
        global session_manager
        if session_manager is None:
            from .server import initialize_llm_client
            llm_client = initialize_llm_client()
            session_manager = GPUEnhancedSessionManager(llm_client)

        # Create advanced session
        session = await session_manager.create_advanced_session(
            "default", request, advanced_params
        )

        return {
            "session_id": session.id,
            "status": session.status,
            "mode": session.mode,
            "gpu_enabled": session.metadata.get('gpu_enabled', False),
            "advanced_config": {
                "selection_method": selection_method,
                "use_fitness_sharing": use_fitness_sharing,
                "use_crowding": use_crowding,
                "n_subpopulations": n_subpopulations,
                "adaptive_parameters": {
                    "mutation": adaptive_mutation,
                    "crossover": adaptive_crossover
                }
            },
            "gpu_config": {
                "device": session_manager.gpu_config.device,
                "batch_size": gpu_batch_size,
                "mixed_precision": session_manager.gpu_config.use_mixed_precision
            }
        }

    except Exception as e:
        logger.error(f"Error creating advanced session: {e}")
        raise


@mcp_gpu.tool()
async def run_batch_experiments(
    experiments: list[dict[str, Any]],
    generations: int = 20,
    use_gpu: bool = True,
    max_batch_size: int = 200,
    checkpoint_interval: int = 10
) -> dict[str, Any]:
    """Run multiple evolution experiments in batch with GPU acceleration.

    This tool enables efficient parallel processing of multiple experiments,
    maximizing GPU utilization through batched operations.

    Args:
        experiments: List of experiment configurations, each containing:
            - experiment_id: Unique identifier
            - prompt: Target prompt for idea generation
            - population_size: Size of population
            - parameters: Optional genetic algorithm parameters
        generations: Number of generations per experiment
        use_gpu: Enable GPU acceleration
        max_batch_size: Maximum batch size for GPU processing
        checkpoint_interval: Generations between checkpoints

    Returns:
        Batch results including all final populations and performance metrics
    """
    try:
        # Initialize session manager if needed
        global session_manager
        if session_manager is None:
            from .server import initialize_llm_client
            llm_client = initialize_llm_client()
            session_manager = GPUEnhancedSessionManager(llm_client)

        # Create batch configuration
        batch_config = BatchEvolutionConfig(
            n_experiments=len(experiments),
            max_batch_size=max_batch_size,
            checkpoint_interval=checkpoint_interval
        )

        # Convert experiments to ExperimentConfig objects
        experiment_configs = []
        for exp in experiments:
            # Create parameters
            params = AdvancedGeneticParameters(
                population_size=exp.get('population_size', 30)
            )

            # Update with any provided parameters
            if 'parameters' in exp:
                for key, value in exp['parameters'].items():
                    if hasattr(params, key):
                        setattr(params, key, value)

            config = ExperimentConfig(
                experiment_id=exp['experiment_id'],
                population_size=exp.get('population_size', 30),
                target_prompt=exp['prompt'],
                parameters=params,
                metadata=exp.get('metadata', {})
            )
            experiment_configs.append(config)

        # Run batch experiments
        results = await session_manager.run_batch_experiments(
            experiment_configs, generations, batch_config
        )

        # Format results
        formatted_results = {}
        for exp_id, population in results['results'].items():
            formatted_results[exp_id] = {
                "top_ideas": sorted(
                    [{"id": idea.id, "content": idea.content, "fitness": idea.fitness}
                     for idea in population],
                    key=lambda x: x['fitness'],
                    reverse=True
                )[:5],
                "population_size": len(population),
                "best_fitness": max(idea.fitness for idea in population) if population else 0
            }

        return {
            "experiments": formatted_results,
            "summary": results['summary'],
            "gpu_used": session_manager.gpu_config.device != "cpu"
        }

    except Exception as e:
        logger.error(f"Error running batch experiments: {e}")
        raise


@mcp_gpu.tool()
async def get_gpu_status() -> dict[str, Any]:
    """Get current GPU status and memory usage.

    Returns information about GPU availability, memory usage,
    and current processing statistics.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "device": "cpu",
                "message": "GPU not available, using CPU"
            }

        # Get GPU info
        device_props = torch.cuda.get_device_properties(0)
        memory_stats = {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "total_gb": device_props.total_memory / (1024**3),
            "free_gb": (device_props.total_memory - torch.cuda.memory_allocated()) / (1024**3)
        }

        return {
            "gpu_available": True,
            "device": f"cuda:{torch.cuda.current_device()}",
            "gpu_name": device_props.name,
            "compute_capability": f"{device_props.major}.{device_props.minor}",
            "memory": memory_stats,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        }

    except ImportError:
        return {
            "gpu_available": False,
            "device": "cpu",
            "message": "PyTorch not installed, GPU features unavailable"
        }
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return {
            "gpu_available": False,
            "device": "cpu",
            "error": str(e)
        }


# Export server components
__all__ = ['mcp_gpu', 'GPUEnhancedSessionManager', 'create_advanced_session',
           'run_batch_experiments', 'get_gpu_status']
