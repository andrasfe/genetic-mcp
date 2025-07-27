"""Genetic MCP - Model Context Protocol server for genetic algorithm-based idea generation."""

__version__ = "0.1.0"

# Export main components
from .optimization_config import OptimizationConfig
from .session_manager_optimized import OptimizedSessionManager

__all__ = [
    "OptimizationConfig",
    "OptimizedSessionManager",
]
