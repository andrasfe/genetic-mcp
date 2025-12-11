# Adding Genetic MCP Server to Cursor

## Quick Setup

Your Cursor MCP configuration file is located at: `~/.cursor/mcp.json`

## Option 1: Manual Configuration (Recommended)

Edit `~/.cursor/mcp.json` and add the following entry to the `mcpServers` object:

```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "/home/andras/genetic-mcp/.venv/bin/python",
      "args": [
        "-m",
        "genetic_mcp.server"
      ],
      "env": {
        "GENETIC_MCP_TRANSPORT": "stdio",
        "PYTHONPATH": "/home/andras/genetic-mcp"
      }
    }
  }
}
```

**Note:** The server will automatically load environment variables from `/home/andras/genetic-mcp/.env` file, so make sure you have:
- `MODEL` - Required (e.g., `meta-llama/llama-3.2-3b-instruct`)
- `DEFAULT_PROVIDER` - Required (one of: `openai`, `anthropic`, `openrouter`)
- At least one API key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`

## Option 2: Using the Setup Script

Run this command to automatically add the configuration:

```bash
cd /home/andras/genetic-mcp
python3 setup_cursor.py
```

## Option 3: If You Have Installed the Package

If you've installed the package with `uv pip install -e .` or `pip install -e .`, you can use:

```json
{
  "mcpServers": {
    "genetic-mcp": {
      "command": "genetic-mcp",
      "args": [],
      "env": {
        "GENETIC_MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

## After Configuration

1. **Restart Cursor** - Close and reopen Cursor for the changes to take effect
2. **Verify Connection** - The MCP server should appear in Cursor's MCP settings
3. **Check Tools** - You should see all the genetic-mcp tools available (create_session, run_generation, etc.)

## Troubleshooting

### Server Not Appearing
- Make sure the `.env` file exists in `/home/andras/genetic-mcp/`
- Check that `uv` is installed: `which uv`
- Verify the server runs manually: `cd /home/andras/genetic-mcp && uv run python -m genetic_mcp.server`

### Environment Variables Not Loading
- The server loads `.env` automatically when run from the project directory
- If using the installed package method, you may need to set environment variables in the `env` section of `mcp.json`

### Check Logs
- Cursor logs MCP server output in its internal logs
- You can also test the server manually to see if it starts correctly

## Current Configuration

Your current `~/.cursor/mcp.json` has these servers:
- `vulnicheck-local-commented`
- `vulnicheck`
- `hinter`

After adding `genetic-mcp`, you'll have all four servers available.

