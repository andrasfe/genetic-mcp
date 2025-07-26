#!/usr/bin/env python3
"""Test MCP configuration for genetic-mcp."""

import json
import subprocess
import sys

def test_mcp_tools():
    """Test that the MCP server exposes the expected tools."""
    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        },
        "id": 1
    }
    
    # Send list tools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    }
    
    try:
        # Start the server process
        proc = subprocess.Popen(
            ["genetic-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send initialize request
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()
        
        # Read response
        init_response = proc.stdout.readline()
        print(f"Initialize response: {init_response.strip()}")
        
        # Send list tools request
        proc.stdin.write(json.dumps(list_tools_request) + "\n")
        proc.stdin.flush()
        
        # Read response
        tools_response = proc.stdout.readline()
        print(f"Tools response: {tools_response.strip()}")
        
        # Parse and display tools
        try:
            tools_data = json.loads(tools_response)
            if "result" in tools_data and "tools" in tools_data["result"]:
                print("\nAvailable tools:")
                for tool in tools_data["result"]["tools"]:
                    print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            else:
                print("No tools found in response")
        except json.JSONDecodeError as e:
            print(f"Failed to parse tools response: {e}")
        
        # Terminate the process
        proc.terminate()
        proc.wait(timeout=5)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_mcp_tools())