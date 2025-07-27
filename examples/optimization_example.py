"""Example demonstrating the unified optimization API.

This example shows how to use different optimization levels with the genetic MCP server.
"""

import asyncio
import os
from typing import List, Dict, Any

# Ensure the parent directory is in the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_mcp.optimization_config import OptimizationConfig
from genetic_mcp.session_manager_optimized import OptimizedSessionManager
from genetic_mcp.llm_client import MultiModelClient
from genetic_mcp.models import GenerationRequest, EvolutionMode, FitnessWeights


async def demo_basic_usage():
    """Demonstrate basic usage without optimizations."""
    print("\n=== Basic Usage (No Optimizations) ===")
    
    # Create configuration
    config = OptimizationConfig(level="basic")
    print(f"Configuration: {config.to_dict()}")
    
    # This would normally use the server, but we'll show the config
    print("- Standard genetic algorithm")
    print("- No GPU acceleration")
    print("- Fixed parameters")
    print("- Simple selection strategies")


async def demo_enhanced_optimizations():
    """Demonstrate enhanced mathematical optimizations."""
    print("\n=== Enhanced Optimizations ===")
    
    # Create configuration
    config = OptimizationConfig(level="enhanced")
    print(f"Configuration: {config.to_dict()}")
    
    print("\nEnabled features:")
    print("- Adaptive parameter tuning")
    print("- Diversity preservation mechanisms")
    print("- Pareto optimization for multi-objective fitness")
    print("- Advanced selection strategies")
    print("- Early stopping to prevent overfitting")
    
    # Show fine-grained control
    print("\nFine-grained control example:")
    custom_config = OptimizationConfig(
        use_adaptive_parameters=True,
        use_diversity_preservation=True,
        use_pareto_optimization=False,  # Disable specific feature
        diversity_threshold=0.4,  # Custom threshold
        early_stopping_patience=5  # More patience
    )
    print(f"Custom configuration: {custom_config.to_dict()}")


async def demo_gpu_acceleration():
    """Demonstrate GPU acceleration features."""
    print("\n=== GPU Acceleration ===")
    
    # Create configuration
    config = OptimizationConfig(level="gpu")
    print(f"Configuration: {config.to_dict()}")
    
    # Check GPU availability
    if not config.use_gpu:
        print("\nWarning: GPU requested but not available!")
        return
    
    print("\nGPU features:")
    print("- Parallel fitness evaluation on GPU")
    print("- Batch processing of embeddings")
    print("- Mixed precision computation")
    print("- Optimized memory management")
    
    # Show GPU-specific settings
    gpu_config = config.get_gpu_config()
    if gpu_config:
        print(f"\nGPU settings:")
        print(f"- Device: {gpu_config.device}")
        print(f"- Batch size: {gpu_config.batch_size}")
        print(f"- Memory fraction: {gpu_config.memory_fraction}")


async def demo_full_optimization():
    """Demonstrate all optimizations combined."""
    print("\n=== Full Optimization Suite ===")
    
    # Create configuration
    config = OptimizationConfig(level="full")
    print(f"Configuration: {config.to_dict()}")
    
    print("\nAll features enabled:")
    print("- Mathematical optimizations (adaptive, diversity, Pareto)")
    print("- GPU acceleration for large-scale computation")
    print("- Advanced selection and evolution strategies")
    print("- Intelligent early stopping")
    print("- Maximum performance mode")


async def demo_environment_configuration():
    """Demonstrate configuration from environment variables."""
    print("\n=== Environment-based Configuration ===")
    
    # Set some example environment variables
    os.environ["GENETIC_MCP_OPTIMIZATION_LEVEL"] = "enhanced"
    os.environ["GENETIC_MCP_USE_GPU"] = "true"
    os.environ["GENETIC_MCP_MAX_WORKERS"] = "30"
    
    # Create config from environment
    config = OptimizationConfig.from_env()
    print(f"Configuration from environment: {config.to_dict()}")
    
    print("\nEnvironment variables:")
    print("- GENETIC_MCP_OPTIMIZATION_LEVEL: Set optimization preset")
    print("- GENETIC_MCP_USE_GPU: Enable GPU acceleration")
    print("- GENETIC_MCP_USE_ADAPTIVE: Enable adaptive parameters")
    print("- GENETIC_MCP_MAX_WORKERS: Set worker pool size")
    print("- GENETIC_MCP_GPU_DEVICE: Specify GPU device (e.g., cuda:1)")


async def demo_recommendations():
    """Demonstrate automatic recommendations based on problem size."""
    print("\n=== Automatic Recommendations ===")
    
    config = OptimizationConfig()
    
    # Small problem
    small_recommendations = config.get_recommended_settings(population_size=20)
    print(f"\nSmall problem (20 individuals): {small_recommendations}")
    
    # Medium problem
    medium_recommendations = config.get_recommended_settings(population_size=75)
    print(f"Medium problem (75 individuals): {medium_recommendations}")
    
    # Large problem
    large_recommendations = config.get_recommended_settings(population_size=200)
    print(f"Large problem (200 individuals): {large_recommendations}")


async def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print("\n=== Backward Compatibility ===")
    
    config = OptimizationConfig()
    
    # Get coordinator config (for existing code)
    coordinator_config = config.get_coordinator_config()
    print("Coordinator config created for backward compatibility")
    
    # Get GPU config (for existing GPU code)
    gpu_config = config.get_gpu_config()
    print(f"GPU config: {'Created' if gpu_config else 'Not needed (GPU disabled)'}")
    
    print("\nExisting code continues to work without modifications!")


async def main():
    """Run all demonstrations."""
    print("Genetic MCP Optimization API Demonstration")
    print("=" * 50)
    
    # Run demos
    await demo_basic_usage()
    await demo_enhanced_optimizations()
    await demo_gpu_acceleration()
    await demo_full_optimization()
    await demo_environment_configuration()
    await demo_recommendations()
    await demo_backward_compatibility()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- Use OptimizationConfig(level='basic|enhanced|gpu|full') for presets")
    print("- Fine-tune individual settings as needed")
    print("- Configure via environment variables for deployment")
    print("- Backward compatible with existing code")
    print("- Automatic recommendations based on problem size")


if __name__ == "__main__":
    asyncio.run(main())