# Optimization Guide for Genetic MCP

This guide explains how to use the unified optimization API to improve performance and results quality.

## Quick Start

### Basic Usage (No Optimizations)
```python
# Default - no optimizations
config = OptimizationConfig()
```

### Enhanced Mathematical Optimizations
```python
# Enable adaptive parameters, diversity preservation, etc.
config = OptimizationConfig(level="enhanced")
```

### GPU Acceleration
```python
# Enable GPU for large-scale computations
config = OptimizationConfig(level="gpu")
```

### Full Optimization Suite
```python
# Enable all optimizations
config = OptimizationConfig(level="full")
```

## Environment Variables

Configure optimizations via environment:

```bash
# Enable optimizations in the server
export GENETIC_MCP_OPTIMIZATION_ENABLED=true

# Set optimization level
export GENETIC_MCP_OPTIMIZATION_LEVEL=enhanced  # basic, enhanced, gpu, full

# Enable specific features
export GENETIC_MCP_USE_GPU=true
export GENETIC_MCP_USE_ADAPTIVE=true
export GENETIC_MCP_MAX_WORKERS=30
export GENETIC_MCP_GPU_DEVICE=cuda:0
```

## Using with MCP Server

### 1. Enable Optimizations Globally

Start the server with optimizations:
```bash
GENETIC_MCP_OPTIMIZATION_ENABLED=true \
GENETIC_MCP_OPTIMIZATION_LEVEL=enhanced \
python -m genetic_mcp.server
```

### 2. Per-Session Optimization

When creating a session, specify the optimization level:
```python
# Via MCP tool
result = await mcp.call_tool(
    "create_session",
    prompt="Generate innovative ideas for sustainable energy",
    population_size=100,
    optimization_level="gpu"  # Enable GPU for this session
)
```

### 3. Monitor Optimization Performance

Get optimization statistics:
```python
# Global stats
stats = await mcp.call_tool("get_optimization_stats")

# Session-specific report
report = await mcp.call_tool(
    "get_optimization_report",
    session_id="your-session-id"
)
```

## Optimization Levels Explained

### Basic (Default)
- Standard genetic algorithm
- No special optimizations
- Good for small problems (<50 individuals)
- Fastest setup, predictable behavior

### Enhanced
- **Adaptive Parameters**: Mutation and crossover rates adjust dynamically
- **Diversity Preservation**: Prevents premature convergence
- **Pareto Optimization**: Better multi-objective handling
- **Advanced Selection**: Multiple selection strategies
- **Early Stopping**: Prevents overfitting
- Good for medium problems (50-200 individuals)

### GPU
- **Parallel Fitness Evaluation**: Batch processing on GPU
- **Accelerated Embeddings**: Fast similarity computation
- **Mixed Precision**: FP16 for speed, FP32 for accuracy
- **Optimized Memory**: Efficient GPU memory usage
- Good for large problems (>100 individuals)

### Full
- All enhanced optimizations enabled
- GPU acceleration enabled
- Maximum performance mode
- Best for large, complex problems

## Fine-Grained Control

```python
# Custom configuration
config = OptimizationConfig(
    # Mathematical optimizations
    use_adaptive_parameters=True,
    use_diversity_preservation=True,
    use_pareto_optimization=False,  # Disable if not needed
    diversity_threshold=0.4,         # Higher = more diversity
    early_stopping_patience=5,       # Generations before stopping
    
    # GPU settings
    use_gpu=True,
    gpu_device="cuda:1",            # Specific GPU
    gpu_batch_size=128,             # Larger = faster but more memory
    
    # Performance
    max_workers=30,                 # Parallel LLM workers
    cache_embeddings=True           # Cache for speed
)
```

## Recommendations by Problem Size

| Population Size | Recommended Level | Key Settings |
|----------------|-------------------|--------------|
| < 20           | basic            | Default settings |
| 20-50          | basic/enhanced   | Consider adaptive parameters |
| 50-100         | enhanced         | Enable diversity preservation |
| 100-200        | enhanced/gpu     | Consider GPU if available |
| > 200          | gpu/full         | GPU strongly recommended |

## Performance Tips

1. **Start Simple**: Begin with basic and upgrade as needed
2. **Monitor GPU Memory**: Use `get_optimization_stats()` to check usage
3. **Batch Operations**: GPU works best with larger batches
4. **Early Stopping**: Prevents wasted computation
5. **Diversity vs Speed**: Higher diversity = better results but slower

## Backward Compatibility

Existing code continues to work:
- Default behavior unchanged
- Optimizations are opt-in
- Old session manager still available
- Gradual migration path

## Troubleshooting

### GPU Not Available
```python
config = OptimizationConfig(level="gpu")
if not config.use_gpu:
    print("GPU requested but not available, falling back to CPU")
```

### Out of Memory
- Reduce `gpu_batch_size`
- Lower `gpu_memory_fraction`
- Use `enhanced` instead of `gpu`

### Slow Performance
- Check worker pool size
- Enable embedding cache
- Consider GPU acceleration

### Poor Result Quality
- Increase diversity threshold
- Enable Pareto optimization
- Disable early stopping