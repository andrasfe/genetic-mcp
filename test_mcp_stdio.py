#!/usr/bin/env python3
"""Test MCP server stdio communication."""

import json
import subprocess
import os

# Test message - correct format for MCP
test_message = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "0.1.0",
        "capabilities": {
            "tools": {}
        },
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    },
    "id": 1
}

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
    cwd=os.path.dirname(os.path.abspath(__file__))
)

# Send request
request = json.dumps(test_message) + "\n"
proc.stdin.write(request)
proc.stdin.flush()

# Read response
try:
    response = proc.stdout.readline()
    if response:
        print("Response:", response)
        result = json.loads(response)
        if "result" in result:
            print("\nAvailable tools:")
            for tool in result["result"].get("tools", []):
                print(f"  - {tool['name']}: {tool.get('description', '')[:60]}...")
    else:
        print("No response received")
        stderr = proc.stderr.read()
        if stderr:
            print("Stderr:", stderr)
except Exception as e:
    print(f"Error: {e}")
    stderr = proc.stderr.read()
    if stderr:
        print("Stderr:", stderr)

# Cleanup
proc.terminate()
proc.wait()