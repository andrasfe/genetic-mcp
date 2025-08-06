# Genetic Algorithm MCP Server

A Model Context Protocol (MCP) server implementing genetic algorithm-based idea generation using parallel LLM workers, multi-objective fitness evaluation, and evolutionary optimization.

## Features

### Core Capabilities
- **Parallel LLM Workers**: Concurrent idea generation with configurable worker pools
- **Multi-Objective Fitness**: Evaluate ideas on relevance, novelty, and feasibility
- **Genetic Operations**: Selection, crossover, mutation, and elitism strategies
- **GPU Acceleration**: Optional CUDA support for embeddings and fitness evaluation
- **Session Management**: Persistent sessions with automatic cleanup
- **Multi-Model Support**: OpenAI, Anthropic, and OpenRouter LLM integrations
- **Progress Streaming**: Real-time updates for long-running operations
- **Lineage Tracking**: Complete evolution history and parent-child relationships

### Advanced Features (New)
- **Session Persistence**: Complete save/load/resume capability with auto-save every 3 minutes
- **Temperature Variation**: Dynamic temperature control for balanced exploration/exploitation
- **Adaptive Population Size**: Automatically adjusts population based on diversity metrics
- **Memory & Learning System**: Persistent learning from past sessions with parameter optimization
- **Hybrid Selection Strategies**: 7 selection methods with UCB1-based adaptive switching
- **Advanced Crossover Operators**: 10 crossover types including semantic and edge recombination
- **Intelligent Mutation**: 9 mutation strategies with fitness landscape analysis
- **Embedding Providers**: Support for OpenAI, Sentence Transformers, Cohere, Voyage AI
- **Client-Generated Mode**: Support for human-in-the-loop idea generation
- **Claude Evaluation Mode**: Combine algorithmic fitness with Claude's qualitative assessment
- **Advanced Optimization**: Adaptive parameters, Pareto optimization, species preservation

## How the Genetic Algorithm Works

This MCP server implements a sophisticated genetic algorithm that evolves ideas through multiple generations, combining the power of LLMs with evolutionary computation principles.

### Core Concepts

1. **Population**: Each generation consists of multiple ideas (default: 10-50)
2. **Fitness Function**: Multi-objective evaluation scoring each idea
3. **Evolution**: Ideas improve through selection, crossover, and mutation
4. **LLM Integration**: Uses language models for intelligent genetic operations

### The Evolution Process

#### Initial Generation (Gen 0)
- Multiple LLM workers generate diverse initial ideas based on your prompt
- Each idea is evaluated for fitness across three dimensions:
  - **Relevance (40%)**: Semantic similarity to the original prompt
  - **Novelty (30%)**: Uniqueness compared to other ideas
  - **Feasibility (30%)**: Practical implementability

#### Subsequent Generations (Gen 1+)
1. **Parent Selection**: Tournament selection picks high-fitness parents
   - Randomly selects 3 ideas, chooses the best
   - Repeats to find two parents for breeding

2. **Crossover (70% probability)**: LLM-guided idea combination
   ```
   Parent 1: "Sustainable vertical farming"
   Parent 2: "AI-powered crop monitoring"
   Offspring: "AI-monitored vertical farming system with adaptive growth optimization"
   ```

3. **Mutation (10% probability)**: Intelligent modifications
   - **Rephrase**: Reword while preserving meaning
   - **Add**: Introduce new elements
   - **Remove**: Simplify by removing components
   - **Modify**: Alter specific aspects

4. **Elitism**: Top 10% of ideas pass unchanged to next generation

### Example Evolution Flow

```
Prompt: "Innovative solutions for urban agriculture"

Generation 0: 50 random ideas
├── "Rooftop hydroponic gardens" (fitness: 0.6)
├── "Community seed sharing network" (fitness: 0.7)
├── "Smart irrigation systems" (fitness: 0.5)
└── ... 47 more ideas

Generation 1: Best ideas combine
├── "Hydroponic + community sharing" (fitness: 0.8)
├── "Smart rooftop networks" (fitness: 0.75)
└── ... evolved population

Generation 2-5: Further refinement
└── Top idea: "Community-driven rooftop hydroponic networks with 
    smart resource sharing and automated climate control" (fitness: 0.95)
```

### Configuration Parameters

```python
GeneticParameters(
    population_size=10,      # Ideas per generation (default: 10)
    generations=5,           # Evolution cycles
    mutation_rate=0.1,       # 10% mutation chance
    crossover_rate=0.7,      # 70% crossover chance
    elitism_rate=0.1         # Preserve top 10% of ideas
)
```

### Why It Works

1. **Exploration vs Exploitation**: Mutations explore new possibilities while crossover exploits successful patterns
2. **Parallel Diversity**: Multiple workers ensure diverse idea generation
3. **Intelligent Operations**: LLMs understand context, creating meaningful combinations
4. **Multi-objective Optimization**: Balances multiple criteria for well-rounded solutions

## Claude Evaluation Mode

The Claude evaluation mode enhances the genetic algorithm by combining algorithmic fitness scores with Claude's qualitative assessment. This creates a more nuanced selection process that considers both quantitative metrics and human-like judgment.

### How It Works

1. **Enable Evaluation**: Call `enable_claude_evaluation` with desired weight (0-1)
2. **Request Evaluation**: Use `evaluate_ideas` to get unevaluated ideas
3. **Submit Assessments**: Claude evaluates ideas and submits scores via `submit_evaluations`
4. **Combined Fitness**: System combines algorithmic and Claude scores based on weight

### Benefits

- **Qualitative Insights**: Captures nuances that algorithms might miss
- **Context Understanding**: Claude can assess real-world feasibility and impact
- **Flexible Weighting**: Adjust balance between algorithmic and qualitative evaluation
- **Backwards Compatible**: Works seamlessly with existing sessions

### Example Workflow

```python
# 1. Create session normally
session = await mcp.create_session(prompt="Urban transportation solutions")

# 2. Enable Claude evaluation (40% weight)
await mcp.enable_claude_evaluation(session_id, evaluation_weight=0.4)

# 3. Run generation
await mcp.run_generation(session_id)

# 4. Get ideas for evaluation
eval_request = await mcp.evaluate_ideas(session_id, batch_size=10)

# 5. Claude evaluates each idea
evaluations = {}
for idea in eval_request['ideas']:
    evaluations[idea['id']] = {
        "score": 0.85,  # 0-1 score
        "justification": "Innovative approach with clear benefits",
        "strengths": ["Scalable", "User-friendly"],
        "weaknesses": ["High initial cost"]
    }

# 6. Submit evaluations
await mcp.submit_evaluations(session_id, evaluations)

# 7. Continue evolution with enhanced fitness
await mcp.run_generation(session_id)  # Uses combined fitness for selection
```

### Evaluation Criteria

Claude evaluates ideas based on:
- **Relevance**: How well it addresses the original prompt
- **Novelty**: Creative and unique aspects
- **Feasibility**: Practical implementation considerations
- **Potential Impact**: Expected value if implemented

## Architecture

Built by a team of collaborative AI agents:
- Systems architecture with modular design
- Mathematical validation using NSGA-II principles
- GPU optimization for performance
- Simplified Python patterns for maintainability
- Comprehensive QA and testing

## Installation

### Method 1: Quick Install with Claude MCP

```bash
claude mcp add genetic-mcp \
  -e OPENROUTER_API_KEY="your-api-key-here" \
  -e OPENAI_API_KEY="your-oai-api-key-here" \
  -e OPENROUTER_MODEL="meta-llama/llama-3.2-3b-instruct" \
  -e OPENAI_MODEL="gpt-4-turbo-preview" \
  -- uvx --from git+https://github.com/YOUR_USERNAME/genetic-mcp.git genetic-mcp
```

This will automatically configure the server with your API key. The full configuration will be added to `~/.claude/claude_desktop_config.json`.

### Method 2: Local Development Installation

1. **Clone and install the package:**
```bash
git clone https://github.com/YOUR_USERNAME/genetic-mcp.git
cd genetic-mcp
uv pip install -e .  # or: pip install -e .
```

2. **Configure in Claude Desktop:**

Edit `~/.claude/claude_desktop_config.json`:

For running from installed package:
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "genetic-mcp",
      "args": [],
      "env": {
        "OPENROUTER_API_KEY": "your-openrouter-api-key",
        "OPENAI_API_KEY": "your-openai-api-key",
        "OPENROUTER_MODEL": "meta-llama/llama-3.3-8b-instruct",
        "OPENAI_MODEL": "gpt-4-turbo-preview",
        "GENETIC_MCP_DEBUG": "false",
        "GENETIC_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

For running locally with uv:
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "-m",
        "genetic_mcp.server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "your-openrouter-api-key",
        "OPENAI_API_KEY": "your-openai-api-key", 
        "OPENROUTER_MODEL": "meta-llama/llama-3.3-8b-instruct",
        "OPENAI_MODEL": "gpt-4-turbo-preview",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    }
  }
}
```

Note: With local installation, the server will automatically use the `OPENROUTER_API_KEY` from your `.env` file.

### Method 3: Run Without Installation

From the project directory:
```bash
# Using uv (recommended)
uv run genetic-mcp

# Or with Python directly
python -m genetic_mcp.server
```

Then configure Claude Desktop to use the local command:
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/genetic-mcp", "run", "genetic-mcp"],
      "env": {
        "OPENROUTER_API_KEY": "your-openrouter-api-key",
        "OPENAI_API_KEY": "your-openai-api-key",
        "OPENROUTER_MODEL": "meta-llama/llama-3.2-3b-instruct",
        "OPENAI_MODEL": "gpt-4-turbo-preview",
        "GENETIC_MCP_DEBUG": "false",
        "GENETIC_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

## Configuration

### API Keys

Create a `.env` file in the project root:
```bash
# REQUIRED: Default model for idea generation
MODEL=meta-llama/llama-3.2-3b-instruct

# Required API key for LLM generation
OPENROUTER_API_KEY=your-openrouter-api-key

# Optional API keys
ANTHROPIC_API_KEY=your-anthropic-api-key  # Alternative LLM provider
OPENAI_API_KEY=your-openai-api-key        # For OpenAI embeddings
COHERE_API_KEY=your-cohere-api-key        # For Cohere embeddings

# Embedding Configuration  
EMBEDDING_PROVIDER=cohere                  # Options: openai, cohere, sentence-transformer
EMBEDDING_MODEL=embed-english-v3.0         # Model for chosen provider

# Persistence Configuration
GENETIC_MCP_MEMORY_ENABLED=true           # Enable memory system for learning
GENETIC_MCP_MEMORY_DB=genetic_mcp_memory.db  # Database for memory system
```

### Environment Variables

#### Core Configuration
- `MODEL`: **REQUIRED** - Default model for idea generation (e.g., `meta-llama/llama-3.2-3b-instruct`)

#### API Configuration
- `OPENROUTER_API_KEY`: OpenRouter API key (required for LLM generation)
- `ANTHROPIC_API_KEY`: Anthropic API key (optional alternative LLM)
- `OPENAI_API_KEY`: OpenAI API key (optional, for OpenAI embeddings)
- `COHERE_API_KEY`: Cohere API key (optional, for Cohere embeddings)
- `VOYAGE_API_KEY`: Voyage AI API key (optional, for Voyage embeddings)

#### Embedding Configuration
- `EMBEDDING_PROVIDER`: Embedding backend (`openai`, `cohere`, `sentence-transformer`, `voyage`, `dummy`)
- `EMBEDDING_MODEL`: Model for chosen provider
  - Cohere: `embed-english-v3.0`, `embed-multilingual-v3.0`
  - OpenAI: `text-embedding-ada-002`, `text-embedding-3-small`
  - Sentence-Transformer: `all-MiniLM-L6-v2` (local, no API needed)

#### Model Overrides (Optional)
- `OPENROUTER_MODEL`: OpenRouter-specific model override (defaults to MODEL)
- `OPENAI_MODEL`: OpenAI-specific model override (defaults to MODEL)
- `ANTHROPIC_MODEL`: Anthropic-specific model override (defaults to MODEL)

#### System Configuration
- `GENETIC_MCP_TRANSPORT`: Transport mode (`stdio` for MCP, `http` for web)
- `GENETIC_MCP_DEBUG`: Enable debug logging (`true`/`false`)
- `GENETIC_MCP_GPU`: Enable GPU acceleration (`true`/`false`)
- `WORKER_POOL_SIZE`: Number of parallel LLM workers (default: 5)
- `SESSION_TTL_SECONDS`: Session timeout in seconds (default: 3600)
- `GENETIC_MCP_MEMORY_ENABLED`: Enable memory & learning system (`true`/`false`, default: true)
- `GENETIC_MCP_MEMORY_DB`: Path to memory database (default: `genetic_mcp_memory.db`)
- `GENETIC_MCP_OPTIMIZATION_ENABLED`: Enable advanced optimization features (`true`/`false`)
- `GENETIC_MCP_OPTIMIZATION_LEVEL`: Optimization level (`basic`, `enhanced`, `gpu`, `full`)

#### Logging Configuration
- `GENETIC_MCP_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `GENETIC_MCP_LOG_FILE`: Optional log file path for persistent logging

## MCP Tools

### 1. create_session
Create a new genetic algorithm session:
```json
{
  "prompt": "innovative solutions for urban transportation",
  "mode": "iterative",  // "single_pass" or "iterative"
  "population_size": 10,
  "top_k": 5,
  "generations": 5,
  "fitness_weights": {
    "relevance": 0.4,
    "novelty": 0.3,
    "feasibility": 0.3
  },
  "models": ["openrouter", "anthropic"],  // Optional
  "client_generated": false,  // Set to true for client-generated mode
  "optimization_level": "enhanced",  // Optional: "basic", "enhanced", "gpu", "full"
  "adaptive_population": true,  // Enable adaptive population size
  "min_population": 5,
  "max_population": 100,
  "diversity_threshold": 0.3,
  "plateau_generations": 3,
  "use_memory_system": true  // Enable learning from past sessions
}
```

### 2. run_generation
Run the generation process for a session:
```json
{
  "session_id": "session-uuid",
  "top_k": 5
}
```

### 3. inject_ideas (Client-Generated Mode)
Inject client-generated ideas into a session:
```json
{
  "session_id": "session-uuid",
  "ideas": [
    "First innovative idea",
    "Second creative solution",
    "Third unique approach"
  ],
  "generation": 0  // Generation number
}
```

### 4. get_progress
Get progress information for a running session:
```json
{
  "session_id": "session-uuid"
}
```

### 5. get_session
Get detailed session information:
```json
{
  "session_id": "session-uuid",
  "include_ideas": true,
  "ideas_limit": 100,
  "ideas_offset": 0,  // For pagination
  "generation_filter": 2  // Optional: filter by generation
}
```

### 6. set_fitness_weights
Update fitness weights for a session:
```json
{
  "session_id": "session-uuid",
  "relevance": 0.5,
  "novelty": 0.3,
  "feasibility": 0.2
}
```

### 7. get_optimization_stats
Get optimization capabilities and usage statistics:
```json
{}  // No parameters required
```

### 8. evaluate_ideas (Claude Evaluation Mode)
Request Claude to evaluate ideas in a session:
```json
{
  "session_id": "session-uuid",
  "idea_ids": ["idea-1", "idea-2"],  // Optional: specific ideas to evaluate
  "evaluation_batch_size": 10  // Number of ideas per batch
}
```

### 9. submit_evaluations (Claude Evaluation Mode)
Submit Claude's evaluations for ideas:
```json
{
  "session_id": "session-uuid",
  "evaluations": {
    "idea-1": {
      "score": 0.85,
      "justification": "Highly innovative and practical",
      "strengths": ["Scalable", "Cost-effective"],
      "weaknesses": ["Complex implementation"]
    }
  }
}
```

### 10. enable_claude_evaluation
Enable Claude evaluation mode for enhanced fitness calculation:
```json
{
  "session_id": "session-uuid",
  "evaluation_weight": 0.5  // Weight for Claude's evaluation (0-1)
}
```

### 11. get_optimization_report
Get detailed optimization report for a session:
```json
{
  "session_id": "session-uuid"
}
```

### 12. get_memory_stats
Get memory system statistics and status:
```json
{}  // No parameters required
```

### 13. get_category_insights
Get insights for a specific prompt category:
```json
{
  "category": "code_generation",  // or "creative_writing", "business_ideas", etc.
  "days": 30  // Number of days to look back
}
```

### 14. save_session
Save current session state to database:
```json
{
  "session_id": "session-uuid",
  "checkpoint_name": "checkpoint-1"  // Optional: name for checkpoint
}
```

### 15. load_session
Load session details from database:
```json
{
  "session_id": "session-uuid"
}
```

### 16. resume_session
Resume a saved session (load + make active):
```json
{
  "session_id": "session-uuid"
}
```

### 17. list_saved_sessions
List saved sessions with filtering:
```json
{
  "client_id": "optional-client-filter",
  "limit": 50,
  "offset": 0
}
```

## Usage Example

### Standard Mode (LLM-Generated Ideas)
1. Create a session with desired configuration
2. Call `run_generation` to start the genetic algorithm
3. Monitor progress with `get_progress`
4. Retrieve results with `get_session`

### Client-Generated Mode
1. Create a session with `client_generated: true`
2. Start `run_generation` in the background
3. Inject ideas for each generation using `inject_ideas`
4. The algorithm will evaluate and evolve based on your ideas
5. Retrieve results showing the best ideas and their fitness scores

Example workflow:
```python
# Create client-generated session
session = create_session(
    prompt="sustainable urban farming",
    mode="iterative",
    population_size=5,
    generations=3,
    client_generated=True
)

# Start generation (runs async)
generation_task = run_generation(session_id)

# Inject ideas for each generation
inject_ideas(session_id, ideas=["idea1", "idea2", ...], generation=0)
# Wait for evaluation...
inject_ideas(session_id, ideas=["evolved1", "evolved2", ...], generation=1)
# Continue for all generations...

# Get results
results = await generation_task
```

## Testing

```bash
# Run all tests (126+ tests currently passing)
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only  
pytest tests/integration/ -v

# Check test coverage
pytest tests/ --cov=genetic_mcp

# Run linting and type checking
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format
```

## Project Structure

```
genetic_mcp/
├── models.py                    # Pydantic data models (v2)
├── server.py                    # FastMCP server implementation
├── session_manager.py           # Session lifecycle management (with auto-save)
├── persistence_manager.py       # Session persistence & recovery system
├── worker_pool.py               # Async LLM worker orchestration (with temperature variation)
├── genetic_algorithm.py         # Core GA operations
├── genetic_algorithm_optimized.py # Enhanced GA with adaptive strategies
├── fitness.py                   # Multi-objective fitness evaluation
├── fitness_enhanced.py          # Advanced fitness with Pareto optimization
├── llm_client.py                # Multi-model LLM support
├── diversity_manager.py         # Species preservation and diversity
├── optimization_coordinator.py  # Advanced GA orchestration
├── adaptive_population.py       # Dynamic population size management
├── memory_system.py             # Persistent learning & parameter optimization
├── hybrid_selection.py          # Multi-strategy selection with UCB1
├── advanced_crossover.py        # 10 crossover operators with adaptation
├── intelligent_mutation.py      # 9 mutation strategies with learning
├── embedding_providers.py       # Multiple embedding backends
├── gpu_*.py                     # GPU acceleration modules
└── tests/                       # Comprehensive test suite (126+ tests)
```

## Logging

The server includes comprehensive logging to track operations at every step.

### Default Log Output

By default, logs are written to **stderr** (standard error stream):
- **Direct execution**: Logs appear in your terminal
- **Claude Desktop**: Logs are captured by MCP but not shown in the UI
- **No file output** unless explicitly configured

### Log Levels
- **DEBUG**: Detailed information for debugging (worker tasks, fitness calculations)
- **INFO**: General operational information (session creation, generation progress) - **Default level**
- **WARNING**: Warning messages (failed tasks, missing embeddings)
- **ERROR**: Error messages with full context
- **CRITICAL**: Critical failures

### Structured Logging
Each component logs with structured context:
- **MCP Tool Calls**: All tool invocations with parameters and execution time
- **Session Lifecycle**: Creation, deletion, and state transitions
- **Worker Pool**: Task distribution, success/failure rates, performance metrics
- **Genetic Algorithm**: Generation creation, selection methods, crossover/mutation operations
- **Fitness Evaluation**: Population statistics, individual fitness scores

### Configuring Logging

#### For Testing/Development
```bash
# Run with debug logging in terminal
GENETIC_MCP_LOG_LEVEL=DEBUG genetic-mcp

# Save logs to file
GENETIC_MCP_LOG_FILE=./genetic_mcp.log genetic-mcp
```

#### For Claude Desktop
Add to `~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "genetic-mcp",
      "env": {
        "GENETIC_MCP_LOG_LEVEL": "INFO",
        "GENETIC_MCP_LOG_FILE": "~/.genetic_mcp/server.log"
      }
    }
  }
}
```

Then view logs with:
```bash
tail -f ~/.genetic_mcp/server.log
```

#### Finding Claude Desktop Logs
When file logging is not configured, check Claude's internal logs:
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%APPDATA%\Claude\logs\`
- **Linux**: `~/.config/Claude/logs/`

### Log Output Examples
```
15:23:45 - genetic_mcp.server - INFO - [CREATE_SESSION] client_id=default mode=iterative population_size=10 client_generated=False
15:23:45 - genetic_mcp.server - INFO - [CREATE_SESSION] duration=0.023s session_id=abc123 client_id=default mode=iterative
15:23:46 - genetic_mcp.session_manager - INFO - Starting generation for session abc123, mode=iterative, population_size=10, generations=5
15:23:47 - genetic_mcp.worker_pool - DEBUG - Worker w1 (openai) processing task t1
15:23:48 - genetic_mcp.worker_pool - INFO - [WORKER_TASK] duration=1.234s worker_id=w1 model=openai task_id=t1 status=success
15:23:52 - genetic_mcp.fitness - INFO - [EVALUATE_POPULATION] duration=0.567s population_size=10 avg_fitness=0.75 max_fitness=0.92
```

## Troubleshooting

### Common Issues

1. **MCP installation fails with uvx**
   - Use local installation method instead (Method 1)
   - Ensure you're in the correct directory when running `uv pip install -e .`

2. **"Command not found: genetic-mcp"**
   - Verify installation: `which genetic-mcp`
   - Check your Python environment is activated
   - Try running with `python -m genetic_mcp.server` instead

3. **OpenRouter API key errors**
   - Ensure `.env` file exists in project root
   - Check API key is valid and has credits
   - Verify key format: `OPENROUTER_API_KEY=sk-or-v1-...`

4. **MCP server not appearing in Claude Desktop**
   - Restart Claude Desktop after editing config
   - Check `~/.claude/claude_desktop_config.json` syntax
   - Look for errors in Claude Desktop logs

5. **"Failed to validate request" errors**
   - This is normal during initialization
   - The server needs proper MCP handshake before accepting tool calls

6. **"OpenAI API key is required for embeddings" error**
   - The system requires OpenAI API key for semantic embeddings
   - Set `OPENAI_API_KEY` in your `.env` file or environment
   - This is required even if you're using other LLMs for idea generation
   - Embeddings are essential for accurate fitness evaluation

### Debug Mode

Enable debug logging to troubleshoot issues:
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "genetic-mcp",
      "args": [],
      "env": {
        "GENETIC_MCP_DEBUG": "true"
      }
    }
  }
}
```

## Documentation

- `ARCHITECTURE.md`: Complete system architecture
- `IMPLEMENTATION_GUIDE.md`: Implementation details
- `DATA_MODELS.md`: Data model specifications
- `SYSTEM_SUMMARY.md`: System overview and insights

## License

MIT