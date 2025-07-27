"""Test script to verify optimization integration works correctly."""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_mcp import OptimizationConfig, OptimizedSessionManager


def test_optimization_config():
    """Test OptimizationConfig creation and presets."""
    print("Testing OptimizationConfig...")
    
    # Test basic config
    basic = OptimizationConfig()
    assert basic.level == "basic"
    assert not basic.use_adaptive_parameters
    assert not basic.use_gpu
    print("✓ Basic config works")
    
    # Test enhanced preset
    enhanced = OptimizationConfig(level="enhanced")
    assert enhanced.use_adaptive_parameters
    assert enhanced.use_diversity_preservation
    assert enhanced.use_pareto_optimization
    print("✓ Enhanced preset works")
    
    # Test GPU preset
    gpu = OptimizationConfig(level="gpu")
    # Note: GPU might be disabled if not available
    assert gpu.level == "gpu"
    print("✓ GPU preset works")
    
    # Test full preset
    full = OptimizationConfig(level="full")
    assert full.use_adaptive_parameters
    assert full.use_diversity_preservation
    print("✓ Full preset works")
    
    # Test custom config
    custom = OptimizationConfig(
        use_adaptive_parameters=True,
        use_gpu=False,
        diversity_threshold=0.5
    )
    assert custom.use_adaptive_parameters
    assert not custom.use_gpu
    assert custom.diversity_threshold == 0.5
    print("✓ Custom config works")
    
    # Test to_dict
    config_dict = custom.to_dict()
    assert isinstance(config_dict, dict)
    assert "mathematical_optimizations" in config_dict
    print("✓ to_dict() works")
    
    print("All OptimizationConfig tests passed!")


def test_environment_config():
    """Test configuration from environment variables."""
    print("\nTesting environment configuration...")
    
    # Set test environment variables
    os.environ["GENETIC_MCP_OPTIMIZATION_LEVEL"] = "enhanced"
    os.environ["GENETIC_MCP_USE_GPU"] = "false"
    os.environ["GENETIC_MCP_MAX_WORKERS"] = "25"
    
    # Create config from environment
    config = OptimizationConfig.from_env()
    assert config.level == "enhanced"
    assert config.use_adaptive_parameters  # Should be set by enhanced preset
    assert not config.use_gpu  # Explicitly disabled
    assert config.max_workers == 25
    print("✓ Environment configuration works")
    
    # Clean up
    del os.environ["GENETIC_MCP_OPTIMIZATION_LEVEL"]
    del os.environ["GENETIC_MCP_USE_GPU"]
    del os.environ["GENETIC_MCP_MAX_WORKERS"]


def test_backward_compatibility():
    """Test backward compatibility methods."""
    print("\nTesting backward compatibility...")
    
    config = OptimizationConfig(level="enhanced")
    
    # Test coordinator config
    coord_config = config.get_coordinator_config()
    assert coord_config.use_adaptive_parameters == config.use_adaptive_parameters
    assert coord_config.use_diversity_preservation == config.use_diversity_preservation
    print("✓ get_coordinator_config() works")
    
    # Test GPU config
    gpu_config = config.get_gpu_config()
    assert gpu_config is None  # Should be None when GPU not enabled
    
    gpu_enabled = OptimizationConfig(use_gpu=True)
    gpu_config = gpu_enabled.get_gpu_config()
    if gpu_enabled.use_gpu:  # Only if GPU is actually available
        assert gpu_config is not None
        assert hasattr(gpu_config, 'device')
    print("✓ get_gpu_config() works")
    
    # Test should_use_optimization_coordinator
    assert config.should_use_optimization_coordinator()
    basic = OptimizationConfig()
    assert not basic.should_use_optimization_coordinator()
    print("✓ should_use_optimization_coordinator() works")


def test_recommendations():
    """Test recommendation system."""
    print("\nTesting recommendations...")
    
    config = OptimizationConfig()
    
    # Small population
    small_rec = config.get_recommended_settings(20)
    assert isinstance(small_rec, dict)
    assert len(small_rec) == 0 or "use_gpu" not in small_rec
    print("✓ Small population recommendations work")
    
    # Large population
    large_rec = config.get_recommended_settings(150)
    assert isinstance(large_rec, dict)
    assert "use_gpu" in large_rec
    assert large_rec["use_gpu"] is True
    print("✓ Large population recommendations work")


async def test_imports():
    """Test that all imports work correctly."""
    print("\nTesting imports...")
    
    # These should not raise errors
    from genetic_mcp import OptimizationConfig, OptimizedSessionManager
    from genetic_mcp.optimization_config import OptimizationConfig as OC
    from genetic_mcp.session_manager_optimized import OptimizedSessionManager as OSM
    
    assert OptimizationConfig is OC
    assert OptimizedSessionManager is OSM
    print("✓ All imports work correctly")


def main():
    """Run all tests."""
    print("Optimization Integration Tests")
    print("=" * 50)
    
    try:
        test_optimization_config()
        test_environment_config()
        test_backward_compatibility()
        test_recommendations()
        asyncio.run(test_imports())
        
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
        print("\nThe optimization integration is working correctly.")
        print("You can now use:")
        print("  - OptimizationConfig for configuration")
        print("  - OptimizedSessionManager for running optimized sessions")
        print("  - Environment variables for deployment configuration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()