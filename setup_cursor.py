#!/usr/bin/env python3
"""Script to add genetic-mcp server to Cursor's MCP configuration."""
import json
import os
from pathlib import Path

def main():
    """Add genetic-mcp to Cursor MCP configuration."""
    config_path = Path.home() / ".cursor" / "mcp.json"
    
    # Read existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}
    
    # Add genetic-mcp server
    project_dir = "/home/andras/genetic-mcp"
    venv_python = f"{project_dir}/.venv/bin/python"
    
    config["mcpServers"]["genetic-mcp"] = {
        "command": venv_python,
        "args": [
            "-m",
            "genetic_mcp.server"
        ],
        "env": {
            "GENETIC_MCP_TRANSPORT": "stdio",
            "PYTHONPATH": project_dir
        }
    }
    
    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Successfully added genetic-mcp to {config_path}")
    print("\n📋 Configuration added:")
    print(json.dumps({"genetic-mcp": config["mcpServers"]["genetic-mcp"]}, indent=2))
    print("\n⚠️  Please restart Cursor for the changes to take effect.")
    print("\n💡 Make sure your .env file is configured with:")
    print("   - MODEL (required)")
    print("   - DEFAULT_PROVIDER (required)")
    print("   - At least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY)")

if __name__ == "__main__":
    main()

