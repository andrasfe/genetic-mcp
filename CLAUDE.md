# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genetic MCP is a Model Context Protocol (MCP) server implementing genetic algorithm-based idea generation using parallel LLM workers, multi-objective fitness evaluation, and evolutionary optimization. It enables AI-powered creative problem solving through evolutionary computation.

## New Memory & Learning System

The genetic-mcp now includes a comprehensive Memory & Learning System that persistently learns from successful evolution patterns and provides intelligent parameter recommendations for new sessions.

### Memory System Features

- **Automatic Prompt Categorization**: Intelligently categorizes prompts (code_generation, creative_writing, business_ideas, problem_solving, research_analysis, design_concepts)
- **Parameter Optimization**: Learns optimal genetic algorithm parameters based on historical performance
- **Success Pattern Storage**: Stores successful evolution patterns in SQLite database
- **Performance Tracking**: Tracks convergence speed, diversity maintenance, and fitness scores
- **Transfer Learning**: Applies learnings from similar past prompts to new sessions
- **Embedding-Based Similarity**: Uses semantic similarity to match prompts with historical data

### Environment Variables

- `GENETIC_MCP_MEMORY_ENABLED`: Enable/disable memory system (default: true)
- `GENETIC_MCP_MEMORY_DB`: Path to SQLite database (default: genetic_mcp_memory.db)

### New MCP Tools

- `get_memory_stats`: Get current memory system status and statistics
- `get_category_insights`: Get insights and recommendations for specific prompt categories

### Memory System Integration

When creating sessions, the memory system:
1. Automatically categorizes the input prompt
2. Searches for similar historical sessions using embedding similarity
3. Recommends optimal parameters based on past performance
4. Uses recommendations if confidence > 70%, otherwise falls back to category defaults
5. Stores session results after completion for future learning

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

# Run without installation
uv run genetic-mcp
python -m genetic_mcp.server
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **MCP Server** (`server.py`): FastMCP-based server implementing 8 genetic algorithm tools
   - `create_session`: Initialize GA session with configuration (supports client-generated mode)
   - `run_generation`: Execute the complete generation process
   - `inject_ideas`: Inject client-generated ideas into a session (client-generated mode only)
   - `get_progress`: Monitor session progress in real-time
   - `get_session`: Retrieve detailed session information with pagination
   - `set_fitness_weights`: Dynamically adjust fitness evaluation weights
   - `get_optimization_stats`: Get optimization capabilities and usage statistics
   - `get_optimization_report`: Detailed optimization report for a session

2. **Session Manager** (`session_manager.py`): Manages GA session lifecycle
   - Thread-safe session storage with TTL (1 hour default)
   - Automatic cleanup of expired sessions
   - Session state transitions and validation
   - Integration with memory system for parameter optimization
   - Automatic storage of session results for learning

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

7. **Embedding Providers** (`embedding_providers.py`): Flexible embedding backend support
   - **OpenAI**: text-embedding-ada-002, text-embedding-3-small/large
   - **Sentence Transformers**: Local models (all-MiniLM-L6-v2, etc.)
   - **Cohere**: embed-english-v3.0 and other models
   - **Voyage AI**: voyage-2 embeddings
   - **Dummy Provider**: Random embeddings for testing
   - Factory pattern for easy provider switching
   - Automatic fallback: Sentence Transformers → OpenAI → Error

8. **GPU Acceleration** (optional `gpu_*.py` modules):
   - CUDA-accelerated embeddings
   - Parallel fitness computation
   - Automatic CPU fallback

9. **Diversity Manager** (`diversity_manager.py`): Preserves population diversity
   - Species clustering using DBSCAN
   - Diversity metrics (Simpson, Shannon, coverage)
   - Niche-based crowding control
   - Species tracking and representatives

10. **Enhanced Fitness Evaluator** (`fitness_enhanced.py`): Advanced fitness features
    - Pareto dominance checking
    - Multi-objective ranking (NSGA-II style)
    - Dynamic weight adjustment
    - Fitness landscape analysis

11. **Optimization Coordinator** (`optimization_coordinator.py`): Advanced GA orchestration
    - Adaptive parameter tuning
    - Multiple selection strategies (tournament, Boltzmann, rank-based)
    - Early stopping with patience
    - Generation statistics tracking
    - Strategy adaptation based on metrics

12. **Hybrid Selection Manager** (`hybrid_selection.py`): Advanced selection strategy system
    - Multi-armed bandit approach with UCB1 algorithm for adaptive strategy selection
    - Seven selection strategies: tournament, roulette wheel, roulette with sigma scaling, truncation, rank-based, stochastic universal sampling, and Boltzmann
    - Performance tracking for each strategy (convergence speed, diversity impact, selection intensity)
    - Adaptive selection pressure based on population state and generation
    - Manual strategy override capability for fine-grained control
    - Strategy-specific parameter tuning (tournament size, truncation percentage, selection pressure)
    - Comprehensive performance reporting and strategy recommendations

13. **Advanced Crossover Operators** (`advanced_crossover.py`): Sophisticated crossover methods
    - 10 different crossover operators: semantic, multi-point, uniform, edge recombination, order-based, blend, concept mapping, syntactic, hierarchical, and adaptive
    - Intelligent adaptive operator selection using Upper Confidence Bound (UCB) algorithm
    - LLM-guided semantic crossover with enhanced prompting and concept analysis
    - Content similarity analysis and complexity estimation for operator selection
    - Performance tracking with usage statistics, success rates, and fitness improvements
    - Fitness-weighted blending and concept mapping for innovative combinations
    - Grammar-aware and structure-preserving crossover methods
    - Fallback mechanisms when advanced operators fail

14. **Optimized Genetic Algorithm** (`genetic_algorithm_optimized.py`): Enhanced GA implementation
    - Adaptive selection strategies
    - Dynamic operator rates
    - Species-based evolution
    - Island model support
    - Integration with advanced crossover operators
    - Crossover performance tracking and reporting

15. **Memory & Learning System** (`memory_system.py`): Persistent learning across sessions
    - Automatic prompt categorization using keyword analysis
    - SQLite database for storing evolution patterns and success metrics
    - Parameter recommendation engine based on historical performance
    - Embedding-based similarity matching for prompt analysis
    - Transfer learning from successful sessions to new ones
    - Operation effectiveness tracking for genetic operators
    - Category-specific insights and performance statistics

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
- Session state validation ensures proper workflow

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

### LLM Provider Configuration
- `OPENROUTER_API_KEY`: Required for LLM generation (configured in .env)
- `ANTHROPIC_API_KEY`: Optional alternative LLM provider
- `OPENROUTER_MODEL`: OpenRouter model to use (default: meta-llama/llama-3.2-3b-instruct)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4-turbo-preview)
- `ANTHROPIC_MODEL`: Anthropic model to use (default: claude-3-opus-20240229)

### Embedding Provider Configuration
- `EMBEDDING_PROVIDER`: Choose embedding backend (openai/sentence-transformer/cohere/voyage/dummy)
- `OPENAI_API_KEY`: For OpenAI embeddings (optional if using other providers)
- `COHERE_API_KEY`: For Cohere embeddings (optional)
- `VOYAGE_API_KEY`: For Voyage AI embeddings (optional)
- `EMBEDDING_MODEL`: Model name for the chosen provider (provider-specific defaults apply)

### Server Configuration
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
  "population_size": 10,
  "generations": 3,
  "top_k": 5,
  "fitness_weights": {
    "relevance": 0.4,
    "novelty": 0.3,
    "feasibility": 0.3
  }
}
```

### Embedding Provider Configuration

The system supports multiple embedding providers with automatic fallback:

1. **Default Priority** (automatic selection):
   - First tries: Sentence Transformers (local, no API needed)
   - Falls back to: OpenAI (if API key available)
   - Final fallback: Error if neither available

2. **Manual Provider Selection**:
   ```bash
   # Use local Sentence Transformers (no API key needed)
   export EMBEDDING_PROVIDER=sentence-transformer
   
   # Use OpenAI embeddings
   export EMBEDDING_PROVIDER=openai
   export OPENAI_API_KEY=your-key
   
   # Use Cohere embeddings
   export EMBEDDING_PROVIDER=cohere
   export COHERE_API_KEY=your-key
   
   # Use Voyage AI embeddings
   export EMBEDDING_PROVIDER=voyage
   export VOYAGE_API_KEY=your-key
   
   # Use dummy embeddings for testing
   export EMBEDDING_PROVIDER=dummy
   ```

3. **Installation for Local Embeddings**:
   ```bash
   # Install Sentence Transformers for free local embeddings
   pip install sentence-transformers
   ```

4. **Impact of Missing Embeddings**:
   - Without proper embeddings, fitness evaluation is significantly degraded
   - The system can still run but idea quality assessment will be random
   - Recommended: Install sentence-transformers or configure an API provider

## Example Usage

Example scripts in the `examples/` directory:
- `mcp_client_example.py`: Basic MCP client demonstrating all server tools
- `client_generated_example.py`: Example of client-generated mode with Claude
- `gpu_accelerated_example.py`: GPU-accelerated fitness computation example
- `test_integration.py`: Integration testing examples

## Quick Start

1. Install with development dependencies:
   ```bash
   make install-dev
   ```

2. Set up embeddings (choose one):
   ```bash
   # Option A: Install local embeddings (recommended - no API needed)
   pip install sentence-transformers
   
   # Option B: Use OpenAI embeddings
   echo "OPENAI_API_KEY=your-openai-key" >> .env
   ```

3. Set up LLM provider in `.env`:
   ```bash
   OPENROUTER_API_KEY=your-openrouter-key
   # Or use other providers:
   # ANTHROPIC_API_KEY=your-anthropic-key
   ```

4. Run tests to verify setup:
   ```bash
   make test
   ```

5. Start the server:
   ```bash
   make run
   ```

## Hybrid Selection Strategies

The genetic-mcp server includes an advanced hybrid selection system that uses a multi-armed bandit approach to automatically choose the best selection strategy for each generation. This system significantly improves evolutionary performance by adapting to the current population state.

### Available Selection Strategies

1. **Tournament Selection**: Selects parents through tournaments with configurable tournament size
   - Best for: Maintaining diversity while ensuring selection pressure
   - Adaptive features: Tournament size adjusts based on population diversity

2. **Roulette Wheel Selection**: Fitness-proportionate selection based on raw fitness values
   - Best for: Early generations with high fitness variance
   - Adaptive features: Automatic fallback for zero-fitness populations

3. **Roulette with Sigma Scaling**: Enhanced roulette wheel with sigma scaling to prevent premature convergence
   - Best for: Populations with low fitness variance
   - Adaptive features: Configurable sigma scaling factor

4. **Truncation Selection**: Selects from the top percentage of the population
   - Best for: Aggressive optimization and exploitation
   - Adaptive features: Truncation percentage adjusts based on diversity

5. **Rank-Based Selection**: Uses fitness ranks instead of raw fitness values
   - Best for: Populations with extreme fitness differences
   - Adaptive features: Selection pressure adapts to population state

6. **Stochastic Universal Sampling (SUS)**: Reduces selection bias through even sampling
   - Best for: Maintaining population diversity
   - Features: Configurable number of pointers

7. **Boltzmann Selection**: Temperature-based selection with annealing
   - Best for: Late-stage optimization with fine-tuning
   - Adaptive features: Temperature annealing over generations

### Using Hybrid Selection

#### Enable in Session Creation
```json
{
  "prompt": "Generate innovative product ideas",
  "hybrid_selection_enabled": true,
  "selection_strategy": "adaptive",  // Let system choose automatically
  "selection_adaptation_window": 5,  // Track performance over 5 generations
  "selection_exploration_constant": 2.0  // UCB1 exploration parameter
}
```

#### Manual Strategy Override
```json
{
  "prompt": "Generate ideas with specific focus",
  "hybrid_selection_enabled": true,
  "selection_strategy": "tournament",  // Force specific strategy
  "selection_adaptation_window": 3
}
```

#### Configuration Options
- `hybrid_selection_enabled`: Enable hybrid selection system (default: false)
- `selection_strategy`: Manual override ("tournament", "roulette_wheel", etc.) or "adaptive"
- `selection_adaptation_window`: Generations to track for performance (default: 5)
- `selection_exploration_constant`: UCB1 exploration parameter (default: 2.0, higher = more exploration)

### Performance Metrics Tracked

The hybrid selection system tracks comprehensive metrics for each strategy:

1. **Selection Intensity**: How aggressively it selects high-fitness individuals
2. **Diversity Preservation**: Ability to maintain population diversity
3. **Convergence Speed**: How quickly it leads to fitness improvements
4. **Exploration vs Exploitation Balance**: Strategy's exploration/exploitation trade-off
5. **Success Rate**: Percentage of times the strategy led to improvements
6. **UCB1 Value**: Multi-armed bandit confidence score

### Multi-Armed Bandit Algorithm

The system uses the UCB1 (Upper Confidence Bound) algorithm to balance exploration and exploitation:

```
UCB1(strategy) = average_reward(strategy) + c * sqrt(log(total_selections) / times_used(strategy))
```

Where:
- `average_reward`: Weighted combination of fitness improvement, diversity preservation, and convergence speed
- `c`: Exploration constant (configurable)
- The strategy with the highest UCB1 value is selected

### Adaptive Features

1. **Strategy-Specific Parameter Tuning**: Each strategy has parameters that adapt based on population state
   - Tournament size adjusts based on diversity
   - Truncation percentage changes with population convergence
   - Selection pressure adapts to fitness landscape

2. **Population State Analysis**: The system analyzes:
   - Population diversity (phenotypic and genotypic)
   - Fitness variance and distribution
   - Convergence patterns
   - Generation progress

3. **Performance-Based Adaptation**: Strategies are chosen based on:
   - Historical performance in similar conditions
   - Current population characteristics
   - Generation stage (early/middle/late)
   - Stagnation detection

### Getting Performance Reports

Use the genetic algorithm's built-in methods to get detailed performance reports:

```python
# Get comprehensive performance report
report = genetic_algorithm.get_hybrid_selection_report()

# Get strategy recommendations for current population
recommendations = genetic_algorithm.get_selection_recommendations(population)

# Set manual override for specific strategy
success = genetic_algorithm.set_manual_selection_override("tournament", generations=3)
```

### Best Practices

1. **Enable for Complex Problems**: Use hybrid selection for complex optimization problems with multiple objectives
2. **Allow Initial Exploration**: Don't set manual overrides too early; let the system explore first
3. **Monitor Performance**: Review performance reports to understand which strategies work best for your specific problems
4. **Tune Exploration Constant**: Higher values (2.5-3.0) for more exploration, lower values (1.0-1.5) for more exploitation
5. **Consider Population Size**: Larger populations benefit more from adaptive selection

## Advanced Crossover Operators

The genetic-mcp server includes sophisticated crossover operators that go beyond basic semantic crossover. These advanced operators use intelligent selection algorithms, content analysis, and LLM-guided recombination to create more innovative and diverse offspring.

### Available Crossover Operators

1. **Semantic Crossover**: LLM-guided semantic blending with enhanced prompting
   - Preserves core concepts while creating innovative combinations
   - Uses context-aware temperature adjustment for optimal creativity
   - Includes fallback mechanisms for robustness

2. **Multi-Point Crossover**: Splits ideas at multiple semantic boundaries
   - Intelligently selects crossover points based on content structure
   - Preserves coherence through post-processing
   - Configurable number of crossover points

3. **Uniform Crossover**: Uses learned masks to determine parent contributions
   - Adaptive masking based on fitness-weighted selection
   - Considers content quality indicators for better results
   - Maintains semantic integrity across selections

4. **Edge Recombination**: Preserves concept relationships and connections
   - Extracts concept graphs from parent ideas
   - Maintains adjacency relationships during recombination
   - Generates coherent offspring respecting concept dependencies

5. **Order-Based Crossover**: Maintains sequential and logical flow
   - Preserves ordering of key elements and steps
   - Handles numbered lists, procedures, and sequential concepts
   - Reconstructs content with logical transitions

6. **Blend Crossover**: Fitness-weighted combination of parent ideas
   - Uses parent fitness scores to determine blending ratios
   - Supports configurable alpha parameters for fine-tuning
   - Creates offspring that inherit from stronger parents

7. **Concept Mapping**: Maps and recombines concepts between parents
   - Identifies similar and complementary concepts
   - Creates novel connections between previously separate ideas
   - Uses LLM guidance for intelligent concept bridging

8. **Syntactic Crossover**: Grammar-aware crossover preserving structure
   - Analyzes and preserves syntactic patterns
   - Maintains grammatical correctness during recombination
   - Structure-aware fallbacks when advanced parsing fails

9. **Hierarchical Crossover**: Tree-based structure crossover for complex ideas
   - Identifies main themes and sub-themes
   - Exchanges subtrees while maintaining logical relationships
   - Preserves hierarchical organization and parent-child relationships

10. **Adaptive Crossover**: Dynamically selects the best operator
    - Uses Upper Confidence Bound (UCB) algorithm for operator selection
    - Analyzes parent characteristics (similarity, complexity) for intelligent selection
    - Tracks performance metrics to improve future selections

### Adaptive Operator Selection

The system uses intelligent heuristics and machine learning principles to select the most appropriate crossover operator:

**Content Analysis**:
- Calculates semantic similarity between parents
- Estimates content complexity using multiple metrics
- Identifies structural patterns and organization

**Performance-Based Selection**:
- Tracks usage statistics and success rates for each operator
- Measures average fitness improvements
- Uses UCB algorithm for exploration-exploitation balance

**Context-Aware Selection**:
- Early generations favor exploratory operators
- Later generations focus on exploitation and refinement
- Considers population diversity and stagnation metrics

### Configuration Options

Enable advanced crossover in session creation:

```json
{
  "advanced_crossover_enabled": true,
  "crossover_strategy": "adaptive",  // or specific operator name
  "crossover_adaptation_enabled": true,
  "crossover_config": {
    "exploration_factor": 0.1,
    "min_usage_threshold": 5,
    "num_points": 2,  // for multi_point
    "alpha": 0.3,     // for blend
    "preserve_order": true  // for edge_recombination
  }
}
```

**Available Strategies**:
- `"semantic"`, `"multi_point"`, `"uniform"`, `"edge_recombination"`
- `"order_based"`, `"blend"`, `"concept_mapping"`, `"syntactic"`
- `"hierarchical"`, `"adaptive"` (recommended)

### Performance Tracking

The system tracks comprehensive metrics for each crossover operator:

- **Usage Statistics**: Count of times each operator was used
- **Success Rates**: Percentage of successful crossover operations
- **Fitness Improvements**: Average fitness gain in offspring vs parents
- **Content Metrics**: Similarity scores and diversity measures

### Getting Performance Reports

```python
# Get crossover performance report
report = genetic_algorithm.get_crossover_performance_report()

# Record performance after fitness evaluation
genetic_algorithm.record_crossover_performance(parent1, parent2, offspring)
```

### Best Practices

1. **Start with Adaptive**: Use `"adaptive"` crossover for automatic operator selection
2. **Allow Learning**: Let the system explore different operators before manual overrides
3. **Monitor Performance**: Review crossover reports to understand operator effectiveness
4. **Configure Wisely**: Adjust exploration factor based on problem complexity
5. **Content Matters**: Complex ideas benefit more from advanced operators like concept mapping
6. **Fallback Safety**: Advanced operators include robust fallback mechanisms