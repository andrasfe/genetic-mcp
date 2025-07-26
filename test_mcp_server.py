#!/usr/bin/env python3
"""Test the MCP server is working correctly."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Test the server directly
from genetic_mcp.server import mcp, initialize_server, shutdown_server


async def test_server():
    """Test the MCP server."""
    print("Testing Genetic MCP Server...")
    
    # Initialize server
    await initialize_server()
    
    try:
        # Import the tool functions directly
        from genetic_mcp.server import (
            create_session, run_generation, get_progress, 
            get_session, set_fitness_weights
        )
        
        # Test create_session tool
        print("\n1. Testing create_session tool...")
        result = await create_session(
            prompt="Test prompt for idea generation",
            mode="single_pass",
            population_size=3,
            top_k=2
        )
        print(f"✓ Created session: {result['session_id']}")
        
        # Test run_generation tool
        print("\n2. Testing run_generation tool...")
        generation_result = await run_generation(
            session_id=result["session_id"],
            top_k=2
        )
        print(f"✓ Generated {generation_result['total_ideas_generated']} ideas")
        print(f"✓ Top idea fitness: {generation_result['top_ideas'][0]['fitness']:.3f}")
        
        # Test get_progress tool
        print("\n3. Testing get_progress tool...")
        progress = await get_progress(
            session_id=result["session_id"]
        )
        print(f"✓ Session status: {progress['status']}")
        
        # Test get_session tool
        print("\n4. Testing get_session tool...")
        session_info = await get_session(
            session_id=result["session_id"]
        )
        print(f"✓ Retrieved session with {len(session_info['ideas'])} ideas")
        
        # Test set_fitness_weights tool
        print("\n5. Testing set_fitness_weights tool...")
        weights_result = await set_fitness_weights(
            session_id=result["session_id"],
            relevance=0.6,
            novelty=0.2,
            feasibility=0.2
        )
        print(f"✓ {weights_result['message']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await shutdown_server()


if __name__ == "__main__":
    asyncio.run(test_server())