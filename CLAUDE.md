# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genetic MCP is a Model Context Protocol (MCP) server implementing genetic algorithm-based idea generation using parallel LLM workers, multi-objective fitness evaluation, and evolutionary optimization. It enables AI-powered creative problem solving through evolutionary computation.

## Key Commands

### Development Setup
```bash
# Install development dependencies
make install-dev

# Or with uv directly
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests (42 tests currently passing)
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage
make test-coverage

# Run a single test file
uv run pytest tests/unit/test_fitness.py -v
```

### Code Quality
```bash
# Run all linting checks (ruff + mypy)
make lint

# Auto-fix linting issues
make lint-fix

# Format code with ruff
make format

# Type checking only
make type-check
```

### Running the Server
```bash
# Run server (default stdio mode)
make run

# Run in stdio mode explicitly
make run-stdio

# Run in HTTP mode with SSE
make run-http

# Run with debug logging
make debug
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **MCP Server** (`server.py`): FastMCP-based server implementing 6 genetic algorithm tools
   - `create_session`: Initialize GA session with configuration (supports client-generated mode)
   - `run_generation`: Execute the complete generation process
   - `inject_ideas`: Inject client-generated ideas into a session (client-generated mode only)
   - `get_progress`: Monitor session progress in real-time
   - `get_session`: Retrieve detailed session information with pagination
   - `set_fitness_weights`: Dynamically adjust fitness evaluation weights

2. **Session Manager** (`session_manager.py`): Manages GA session lifecycle
   - Thread-safe session storage with TTL (1 hour default)
   - Automatic cleanup of expired sessions
   - Session state transitions and validation

3. **Worker Pool** (`worker_pool.py`): Orchestrates parallel LLM workers
   - Configurable worker count (default 5)
   - Dynamic load balancing across workers
   - Retry logic with exponential backoff
   - Circuit breaker for failing workers

4. **Genetic Algorithm** (`genetic_algorithm.py`): Core GA operations
   - Tournament selection with configurable size
   - Semantic crossover using LLM guidance
   - Adaptive mutation rates
   - Elite preservation (top 10%)
   - NSGA-II inspired multi-objective optimization

5. **Fitness Evaluator** (`fitness.py`): Multi-objective fitness calculation
   - Relevance: Semantic similarity to prompt
   - Novelty: Diversity from existing ideas
   - Feasibility: Practical implementation assessment
   - Configurable weights for each objective

6. **LLM Client** (`llm_client.py`): Multi-provider LLM integration
   - Supports OpenAI, Anthropic, OpenRouter
   - Model-specific optimizations
   - Token counting and rate limiting
   - Structured output parsing

7. **GPU Acceleration** (optional `gpu_*.py` modules):
   - CUDA-accelerated embeddings
   - Parallel fitness computation
   - Automatic CPU fallback

8. **Diversity Manager** (`diversity_manager.py`): Preserves population diversity
   - Species clustering using DBSCAN
   - Diversity metrics (Simpson, Shannon, coverage)
   - Niche-based crowding control
   - Species tracking and representatives

9. **Enhanced Fitness Evaluator** (`fitness_enhanced.py`): Advanced fitness features
   - Pareto dominance checking
   - Multi-objective ranking (NSGA-II style)
   - Dynamic weight adjustment
   - Fitness landscape analysis

10. **Optimization Coordinator** (`optimization_coordinator.py`): Advanced GA orchestration
    - Adaptive parameter tuning
    - Multiple selection strategies (tournament, Boltzmann, rank-based)
    - Early stopping with patience
    - Generation statistics tracking
    - Strategy adaptation based on metrics

11. **Optimized Genetic Algorithm** (`genetic_algorithm_optimized.py`): Enhanced GA implementation
    - Adaptive selection strategies
    - Dynamic operator rates
    - Species-based evolution
    - Island model support
    - Advanced crossover methods

## Key Implementation Details

### Parallel Processing
- Worker pool uses `asyncio` for concurrent LLM calls
- Each worker maintains its own LLM client instance
- Batch processing with configurable chunk sizes

### Genetic Operations
- **Selection**: Tournament selection preserves diversity
- **Crossover**: Semantic blending using LLM to combine parent ideas
- **Mutation**: Context-aware modifications via LLM
- **Elitism**: Top 10% preserved across generations

### Session Management
- Sessions expire after 1 hour of inactivity
- Background task cleans up expired sessions every 5 minutes
- Thread-safe operations using asyncio locks

### Error Handling
- Comprehensive retry logic for LLM API failures
- Graceful degradation when workers fail
- Detailed error messages in MCP responses

### Advanced Optimization Features
- **Adaptive Parameters**: Mutation and crossover rates adjust based on diversity
- **Multi-Strategy Selection**: Switches between tournament, Boltzmann, and rank-based
- **Species Preservation**: DBSCAN clustering maintains diverse idea niches
- **Pareto Optimization**: True multi-objective optimization with dominance ranking
- **Early Stopping**: Monitors fitness plateau with configurable patience
- **Island Model**: Population subdivisions with periodic migration

## Environment Configuration

Key environment variables:
- `OPENROUTER_API_KEY`: Required for LLM access (configured in .env)
- `OPENROUTER_MODEL`: OpenRouter model to use (default: meta-llama/llama-3.2-3b-instruct)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4-turbo-preview)
- `ANTHROPIC_MODEL`: Anthropic model to use (default: claude-3-opus-20240229)
- `GENETIC_MCP_TRANSPORT`: Transport mode (stdio/http)
- `GENETIC_MCP_DEBUG`: Enable debug logging
- `GENETIC_MCP_GPU`: Enable GPU acceleration
- `WORKER_POOL_SIZE`: Number of parallel workers (default 5)
- `SESSION_TTL_SECONDS`: Session timeout (default 3600)

## Testing Approach

- Unit tests for individual components (fitness, GA operations, models)
- Integration tests for end-to-end workflows
- Async test support with pytest-asyncio
- Mock LLM responses for deterministic testing
- GPU tests automatically skipped if CUDA unavailable
- Client-generated mode tests with mock idea injection

## Client-Generated Mode

The server supports a special mode where the client (e.g., Claude) generates ideas instead of LLM workers:

1. Create session with `client_generated: true`
2. Session initializes without worker pool
3. `run_generation` waits for client to inject ideas
4. Client calls `inject_ideas` for each generation
5. Server evaluates fitness and manages evolution
6. Supports timeout handling and validation

This mode enables:
- Human-in-the-loop genetic algorithms
- Custom idea generation strategies
- Integration with external idea sources
- Reduced LLM API costs for idea generation

## Important Implementation Notes

- Uses Pydantic v2 for data validation
- FastMCP handles protocol compliance and transport
- Semantic embeddings cached for performance
- Tournament selection size affects convergence speed
- Mutation rate adapts based on population diversity
- Worker failures don't block evolution progress

## Troubleshooting

### Client Timeout Issues

If clients disconnect during long-running operations:

1. **Increase client timeout**: Configure your MCP client with longer timeout values
2. **Use smaller populations**: Reduce `population_size` (default 50) to speed up generations
3. **Reduce worker count**: Lower `WORKER_POOL_SIZE` if rate limits are causing delays
4. **Monitor progress**: Use `get_progress` tool to track session status

Example configuration for faster operations:
```json
{
  "population_size": 20,
  "generations": 3,
  "worker_pool_size": 5
}
```

### Embedding Provider Warnings

The warning "No embedding provider available, using dummy embeddings" indicates:
- No OpenAI API key configured (embeddings use OpenAI by default)
- To fix: Set `OPENAI_API_KEY` environment variable
- Impact: Fitness evaluation less accurate without real embeddings
- Workaround: System still functions with random embeddings for testing

## Example Usage

Example scripts in the `examples/` directory:
- `mcp_client_example.py`: Basic MCP client demonstrating all server tools
- `client_generated_example.py`: Example of client-generated mode with Claude
- `gpu_accelerated_example.py`: GPU-accelerated fitness computation example
- `test_integration.py`: Integration testing examples