#!/usr/bin/env python3
"""Example of using Genetic MCP server via stdio transport."""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict, Optional


class StdioMCPClient:
    """Simple MCP client for stdio communication."""
    
    def __init__(self, server_command: str = "genetic-mcp"):
        self.server_command = server_command
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
    
    async def connect(self):
        """Start the MCP server process."""
        self.process = subprocess.Popen(
            [self.server_command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print(f"Started MCP server process (PID: {self.process.pid})")
        
        # Wait for initialization
        await asyncio.sleep(1)
    
    async def disconnect(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Stopped MCP server process")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not self.process:
            raise RuntimeError("Not connected to MCP server")
        
        # Create request
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self.request_id
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read response
        response_str = self.process.stdout.readline()
        if not response_str:
            raise RuntimeError("No response from MCP server")
        
        response = json.loads(response_str)
        
        # Check for error
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        # Extract result
        if "result" in response and "content" in response["result"]:
            content = response["result"]["content"][0]["text"]
            return json.loads(content)
        
        return response.get("result", {})


async def main():
    """Run example MCP client interactions."""
    client = StdioMCPClient()
    
    try:
        # Connect to server
        await client.connect()
        print("\n=== Genetic MCP Client Example ===\n")
        
        # Example 1: Create a single-pass session
        print("1. Creating single-pass session...")
        session1 = await client.call_tool("create_session", {
            "prompt": "Generate creative solutions for urban farming in small spaces",
            "mode": "single_pass",
            "population_size": 5,
            "top_k": 2
        })
        print(f"Created session: {session1['session_id']}")
        print(f"Mode: {session1['mode']}")
        print(f"Population size: {session1['parameters']['population_size']}")
        
        # Run generation
        print("\n2. Running single-pass generation...")
        result1 = await client.call_tool("run_generation", {
            "session_id": session1["session_id"],
            "top_k": 2
        })
        
        print(f"Generated {result1['total_ideas_generated']} ideas in {result1['execution_time_seconds']:.1f}s")
        print("\nTop ideas:")
        for i, idea in enumerate(result1["top_ideas"], 1):
            print(f"\n--- Idea {i} (Fitness: {idea['fitness']:.3f}) ---")
            print(f"{idea['content'][:200]}...")
        
        # Example 2: Create an iterative evolution session
        print("\n\n3. Creating iterative evolution session...")
        session2 = await client.call_tool("create_session", {
            "prompt": "Design innovative ways to combine renewable energy with urban architecture",
            "mode": "iterative",
            "population_size": 4,
            "generations": 2,
            "top_k": 2,
            "fitness_weights": {
                "relevance": 0.5,
                "novelty": 0.3,
                "feasibility": 0.2
            }
        })
        print(f"Created evolution session: {session2['session_id']}")
        print(f"Generations planned: {session2['parameters']['generations']}")
        print(f"Fitness weights: {session2['fitness_weights']}")
        
        # Run evolution
        print("\n4. Running genetic evolution...")
        result2 = await client.call_tool("run_generation", {
            "session_id": session2["session_id"],
            "top_k": 2
        })
        
        print(f"Completed {result2['generations_completed']} generations")
        print(f"Total ideas evolved: {result2['total_ideas_generated']}")
        print(f"Execution time: {result2['execution_time_seconds']:.1f}s")
        
        # Show lineage
        print("\nIdea lineage (sample):")
        for idea_id, parents in list(result2["lineage"].items())[:3]:
            print(f"  {idea_id} <- {parents}")
        
        print("\nTop evolved ideas:")
        for i, idea in enumerate(result2["top_ideas"], 1):
            print(f"\n--- Evolved Idea {i} ---")
            print(f"Generation: {idea['generation']}")
            print(f"Fitness: {idea['fitness']:.3f}")
            print(f"Scores: R={idea['scores']['relevance']:.2f}, "
                  f"N={idea['scores']['novelty']:.2f}, "
                  f"F={idea['scores']['feasibility']:.2f}")
            print(f"Content: {idea['content'][:250]}...")
        
        # Example 3: Get session details
        print("\n\n5. Getting session details...")
        session_info = await client.call_tool("get_session", {
            "session_id": session2["session_id"]
        })
        
        print(f"Session status: {session_info['status']}")
        print(f"Total ideas in session: {len(session_info['ideas'])}")
        print(f"Worker stats: {session_info['worker_stats']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        await client.disconnect()


if __name__ == "__main__":
    # Note: This requires the MCP server to be running
    # You can start it with: genetic-mcp
    asyncio.run(main())