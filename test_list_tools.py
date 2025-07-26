#!/usr/bin/env python3
"""Test listing MCP tools."""

import json
import subprocess
import os
import time

# Start the server
env = os.environ.copy()
# Use environment variable for API key
if "OPENROUTER_API_KEY" not in env:
    print("Warning: OPENROUTER_API_KEY not set in environment")
    env["OPENROUTER_API_KEY"] = "your-api-key-here"

proc = subprocess.Popen(
    ["uv", "run", "genetic-mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env,
    cwd="/home/andras/genetic_mcp"
)

# Initialize first
init_msg = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "0.1.0",
        "capabilities": {"tools": {}},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    },
    "id": 1
}

proc.stdin.write(json.dumps(init_msg) + "\n")
proc.stdin.flush()
init_response = proc.stdout.readline()
print("Init response:", json.loads(init_response))

# Now list tools
list_tools_msg = {
    "jsonrpc": "2.0", 
    "method": "tools/list",
    "params": {},
    "id": 2
}

proc.stdin.write(json.dumps(list_tools_msg) + "\n")
proc.stdin.flush()
tools_response = proc.stdout.readline()

if tools_response:
    result = json.loads(tools_response)
    print("\nTools response:", result)
    
    if "result" in result and "tools" in result["result"]:
        print("\nAvailable Genetic MCP tools:")
        for tool in result["result"]["tools"]:
            print(f"\n- {tool['name']}")
            print(f"  Description: {tool.get('description', 'No description')[:100]}...")
            if "inputSchema" in tool:
                schema = tool["inputSchema"]
                if "properties" in schema:
                    print("  Parameters:")
                    for param, details in schema["properties"].items():
                        print(f"    - {param}: {details.get('description', 'No description')[:80]}...")

# Cleanup
proc.terminate()
proc.wait()