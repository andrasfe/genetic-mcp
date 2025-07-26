# Genetic Algorithm MCP Server

A Model Context Protocol (MCP) server implementing genetic algorithm-based idea generation using parallel LLM workers, multi-objective fitness evaluation, and evolutionary optimization.

## Features

- **Parallel LLM Workers**: Concurrent idea generation with configurable worker pools
- **Multi-Objective Fitness**: Evaluate ideas on relevance, novelty, and feasibility
- **Genetic Operations**: Selection, crossover, mutation, and elitism strategies
- **GPU Acceleration**: Optional CUDA support for embeddings and fitness evaluation
- **Session Management**: Persistent sessions with automatic cleanup
- **Multi-Model Support**: OpenAI, Anthropic, and OpenRouter LLM integrations
- **Progress Streaming**: Real-time updates for long-running operations
- **Lineage Tracking**: Complete evolution history and parent-child relationships

## Architecture

Built by a team of collaborative AI agents:
- Systems architecture with modular design
- Mathematical validation using NSGA-II principles
- GPU optimization for performance
- Simplified Python patterns for maintainability
- Comprehensive QA and testing

## Installation

### Quick Install with Claude Desktop

```bash
claude mcp add genetic-mcp -- uvx --from git+https://github.com/andrasfe/genetic-mcp.git genetic-mcp
```

### Manual Installation

1. **Clone and install:**
```bash
git clone https://github.com/andrasfe/genetic-mcp.git
cd genetic-mcp
pip install -e .
```

2. **Configure in Claude Desktop:**
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "python",
      "args": ["-m", "genetic_mcp.server"],
      "env": {
        "OPENROUTER_API_KEY": "your-api-key",
        "GENETIC_MCP_DEBUG": "false",
        "GENETIC_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

### Run Locally from Current Directory

```bash
# Using uv (recommended)
uv run genetic-mcp

# Or with Python directly
python -m genetic_mcp.server

# Or if installed
genetic-mcp
```

## Configuration

The project includes a `.env` file with OpenRouter API key configured.

Environment variables:
- `OPENROUTER_API_KEY`: OpenRouter API key (configured)
- `GENETIC_MCP_TRANSPORT`: Transport mode (stdio/http)
- `GENETIC_MCP_DEBUG`: Enable debug logging
- `GENETIC_MCP_GPU`: Enable GPU acceleration

## MCP Tools

### 1. create_session
Create a new genetic algorithm session with configuration:
```json
{
  "population_size": 50,
  "max_generations": 20,
  "fitness_weights": {
    "relevance": 0.4,
    "novelty": 0.3,
    "feasibility": 0.3
  }
}
```

### 2. generate
Generate initial population of ideas:
```json
{
  "session_id": "session-uuid",
  "prompt": "innovative solutions for urban transportation",
  "count": 20
}
```

### 3. evolve
Evolve population through genetic operations:
```json
{
  "session_id": "session-uuid",
  "generations": 10
}
```

### 4. get_results
Retrieve top-K ideas with scores and lineage:
```json
{
  "session_id": "session-uuid",
  "top_k": 5,
  "include_lineage": true
}
```

### 5. list_sessions
List all active sessions:
```json
{}
```

## Usage Example

1. Create a session with desired configuration
2. Generate initial population with your prompt
3. Evolve for multiple generations
4. Retrieve top results with fitness scores

## Testing

```bash
# Run all tests (35 tests pass)
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=genetic_mcp

# Run linting
ruff check genetic_mcp/
ruff format genetic_mcp/
```

## Project Structure

```
genetic_mcp/
├── models.py           # Pydantic data models (v2)
├── server.py           # FastMCP server implementation
├── session_manager.py  # Session lifecycle management
├── worker_pool.py      # Async LLM worker orchestration
├── genetic_algorithm.py # GA operations
├── fitness_evaluator.py # Multi-objective fitness
├── llm_client.py       # Multi-model LLM support
├── gpu_*.py            # GPU acceleration modules
└── tests/              # Comprehensive test suite
```

## Documentation

- `ARCHITECTURE.md`: Complete system architecture
- `IMPLEMENTATION_GUIDE.md`: Implementation details
- `DATA_MODELS.md`: Data model specifications
- `SYSTEM_SUMMARY.md`: System overview and insights

## License

MIT