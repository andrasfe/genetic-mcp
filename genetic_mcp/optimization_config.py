"""Unified optimization configuration for genetic algorithm.

This module provides a simple, unified API for managing all optimization settings
while maintaining backward compatibility.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Unified configuration for all genetic algorithm optimizations.

    This class provides a simple interface to control various optimization levels:
    - Basic: Standard genetic algorithm with no optimizations
    - Enhanced: Mathematical optimizations (adaptive parameters, diversity preservation)
    - GPU: Hardware acceleration for large-scale computations
    - Full: All optimizations enabled

    Example:
        # Basic usage - no optimizations
        config = OptimizationConfig()

        # Enable enhanced math optimizations
        config = OptimizationConfig(level="enhanced")

        # Enable GPU acceleration
        config = OptimizationConfig(level="gpu")

        # Enable everything
        config = OptimizationConfig(level="full")

        # Fine-grained control
        config = OptimizationConfig(
            use_adaptive_parameters=True,
            use_gpu=True,
            gpu_device="cuda:0"
        )
    """

    # Optimization level presets
    level: str = "basic"  # basic, enhanced, gpu, full

    # Mathematical optimizations
    use_adaptive_parameters: bool = False
    use_diversity_preservation: bool = False
    use_pareto_optimization: bool = False
    use_advanced_selection: bool = False
    use_llm_operators: bool = True
    use_early_stopping: bool = False
    early_stopping_patience: int = 3
    diversity_threshold: float = 0.3
    target_species: int = 5

    # GPU acceleration
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_batch_size: int = 64
    gpu_memory_fraction: float = 0.8
    use_mixed_precision: bool = False

    # Performance settings
    cache_embeddings: bool = True
    parallel_evaluation: bool = True
    max_workers: int = 20
    worker_timeout_seconds: int = 30

    # Backward compatibility
    selection_strategy: str = "tournament"  # adaptive, tournament, boltzmann, rank, sus

    def __post_init__(self):
        """Apply optimization level presets."""
        if self.level != "basic":
            self._apply_level_preset(self.level)

        # Auto-detect GPU availability if requested
        if self.use_gpu:
            self._validate_gpu_settings()

    def _apply_level_preset(self, level: str):
        """Apply preset configurations based on optimization level."""
        if level == "enhanced":
            # Enable mathematical optimizations
            self.use_adaptive_parameters = True
            self.use_diversity_preservation = True
            self.use_pareto_optimization = True
            self.use_advanced_selection = True
            self.use_early_stopping = True
            self.selection_strategy = "adaptive"
            logger.info("Applied 'enhanced' optimization preset")

        elif level == "gpu":
            # Enable GPU acceleration
            self.use_gpu = True
            self.use_mixed_precision = True
            self.parallel_evaluation = True
            logger.info("Applied 'gpu' optimization preset")

        elif level == "full":
            # Enable all optimizations
            self.use_adaptive_parameters = True
            self.use_diversity_preservation = True
            self.use_pareto_optimization = True
            self.use_advanced_selection = True
            self.use_early_stopping = True
            self.use_gpu = True
            self.use_mixed_precision = True
            self.parallel_evaluation = True
            self.selection_strategy = "adaptive"
            logger.info("Applied 'full' optimization preset")

        else:
            raise ValueError(f"Unknown optimization level: {level}")

    def _validate_gpu_settings(self):
        """Validate and adjust GPU settings based on availability."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("GPU requested but CUDA not available. Disabling GPU acceleration.")
                self.use_gpu = False
                return

            # Check specific device
            if self.gpu_device.startswith("cuda:"):
                device_id = int(self.gpu_device.split(":")[1])
                if device_id >= torch.cuda.device_count():
                    logger.warning(f"GPU device {self.gpu_device} not found. Using cuda:0")
                    self.gpu_device = "cuda:0"

        except ImportError:
            logger.warning("PyTorch not installed. Disabling GPU acceleration.")
            self.use_gpu = False

    @classmethod
    def from_env(cls) -> "OptimizationConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Read level from environment
        level = os.getenv("GENETIC_MCP_OPTIMIZATION_LEVEL", "basic").lower()
        if level != "basic":
            config = cls(level=level)

        # Override specific settings from environment
        if os.getenv("GENETIC_MCP_USE_GPU", "").lower() == "true":
            config.use_gpu = True
            config.gpu_device = os.getenv("GENETIC_MCP_GPU_DEVICE", config.gpu_device)

        if os.getenv("GENETIC_MCP_USE_ADAPTIVE", "").lower() == "true":
            config.use_adaptive_parameters = True

        if os.getenv("GENETIC_MCP_MAX_WORKERS"):
            config.max_workers = int(os.getenv("GENETIC_MCP_MAX_WORKERS"))

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "level": self.level,
            "mathematical_optimizations": {
                "adaptive_parameters": self.use_adaptive_parameters,
                "diversity_preservation": self.use_diversity_preservation,
                "pareto_optimization": self.use_pareto_optimization,
                "advanced_selection": self.use_advanced_selection,
                "early_stopping": self.use_early_stopping,
                "selection_strategy": self.selection_strategy
            },
            "gpu_acceleration": {
                "enabled": self.use_gpu,
                "device": self.gpu_device,
                "batch_size": self.gpu_batch_size,
                "mixed_precision": self.use_mixed_precision
            },
            "performance": {
                "cache_embeddings": self.cache_embeddings,
                "parallel_evaluation": self.parallel_evaluation,
                "max_workers": self.max_workers
            }
        }

    def get_coordinator_config(self):
        """Get configuration for OptimizationCoordinator (backward compatibility)."""
        from .optimization_coordinator import OptimizationConfig as CoordinatorConfig

        return CoordinatorConfig(
            use_adaptive_parameters=self.use_adaptive_parameters,
            use_diversity_preservation=self.use_diversity_preservation,
            use_pareto_optimization=self.use_pareto_optimization,
            use_llm_operators=self.use_llm_operators,
            use_early_stopping=self.use_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            diversity_threshold=self.diversity_threshold,
            target_species=self.target_species,
            selection_strategy=self.selection_strategy
        )

    def get_gpu_config(self) -> Any | None:
        """Get GPU configuration if enabled."""
        if not self.use_gpu:
            return None

        from .gpu_accelerated import GPUConfig

        return GPUConfig(
            device=self.gpu_device,
            batch_size=self.gpu_batch_size,
            use_mixed_precision=self.use_mixed_precision,
            memory_fraction=self.gpu_memory_fraction
        )

    def should_use_optimization_coordinator(self) -> bool:
        """Determine if we should use the OptimizationCoordinator."""
        return (
            self.use_adaptive_parameters or
            self.use_diversity_preservation or
            self.use_pareto_optimization or
            self.use_early_stopping or
            self.use_advanced_selection
        )

    def get_recommended_settings(self, population_size: int) -> dict[str, Any]:
        """Get recommended settings based on problem size."""
        recommendations = {}

        if population_size > 100:
            recommendations["use_gpu"] = True
            recommendations["gpu_batch_size"] = min(128, population_size)
            recommendations["use_early_stopping"] = True

        if population_size > 50:
            recommendations["use_adaptive_parameters"] = True
            recommendations["use_diversity_preservation"] = True

        return recommendations
