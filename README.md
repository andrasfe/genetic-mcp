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

### Method 1: Quick Install with Claude MCP

```bash
claude mcp add genetic-mcp \
  -e OPENROUTER_API_KEY="your-api-key-here" \
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
```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "genetic-mcp",
      "args": [],
      "env": {
        "GENETIC_MCP_DEBUG": "false",
        "GENETIC_MCP_TRANSPORT": "stdio"
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
# Required: At least one API key
OPENROUTER_API_KEY=your-openrouter-api-key
OPENAI_API_KEY=your-openai-api-key        # Optional
ANTHROPIC_API_KEY=your-anthropic-api-key  # Optional

# LLM Model Configuration
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct  # Default model for OpenRouter
OPENAI_MODEL=gpt-4-turbo-preview                   # Default model for OpenAI
ANTHROPIC_MODEL=claude-3-opus-20240229             # Default model for Anthropic
```

### Environment Variables

#### API Configuration
- `OPENROUTER_API_KEY`: OpenRouter API key (supports multiple models)
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)

#### Model Selection
- `OPENROUTER_MODEL`: Model to use with OpenRouter (default: `meta-llama/llama-3.2-3b-instruct`)
  - Examples: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `google/gemini-2.0-flash-thinking-exp-1219:free`
- `OPENAI_MODEL`: Model to use with OpenAI (default: `gpt-4-turbo-preview`)
- `ANTHROPIC_MODEL`: Model to use with Anthropic (default: `claude-3-opus-20240229`)

#### System Configuration
- `GENETIC_MCP_TRANSPORT`: Transport mode (`stdio` for MCP, `http` for web)
- `GENETIC_MCP_DEBUG`: Enable debug logging (`true`/`false`)
- `GENETIC_MCP_GPU`: Enable GPU acceleration (`true`/`false`)
- `WORKER_POOL_SIZE`: Number of parallel LLM workers (default: 5)
- `SESSION_TTL_SECONDS`: Session timeout in seconds (default: 3600)

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