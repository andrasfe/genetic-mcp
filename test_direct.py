#!/usr/bin/env python3
"""Direct test of the genetic algorithm functionality."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import components
from genetic_mcp.llm_client import OpenRouterClient, MultiModelClient
from genetic_mcp.models import GenerationRequest, EvolutionMode
from genetic_mcp.session_manager import SessionManager


async def test_genetic_system():
    """Test the genetic system directly."""
    print("Testing Genetic MCP System Directly...\n")
    
    # Initialize LLM client
    client = MultiModelClient()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return
    
    openrouter_client = OpenRouterClient(api_key)
    client.add_client("openrouter", openrouter_client, is_default=True)
    
    # Create session manager
    manager = SessionManager(client)
    await manager.start()
    
    try:
        # Test 1: Single pass generation
        print("=== Test 1: Single Pass Generation ===")
        request1 = GenerationRequest(
            prompt="Generate creative ideas for a smart city traffic management system",
            mode=EvolutionMode.SINGLE_PASS,
            population_size=4,
            top_k=2
        )
        
        session1 = await manager.create_session("test_client", request1)
        print(f"Created session: {session1.id}")
        
        result1 = await manager.run_generation(session1, top_k=2)
        print(f"Generated {result1.total_ideas_generated} ideas in {result1.execution_time_seconds:.1f}s")
        
        for i, idea in enumerate(result1.top_ideas, 1):
            print(f"\nIdea {i} (Fitness: {idea.fitness:.3f}):")
            print(f"  {idea.content[:150]}...")
        
        # Test 2: Genetic evolution
        print("\n\n=== Test 2: Genetic Evolution ===")
        request2 = GenerationRequest(
            prompt="Design an innovative educational platform that combines VR and AI",
            mode=EvolutionMode.ITERATIVE,
            population_size=4,
            generations=2,
            top_k=2
        )
        
        session2 = await manager.create_session("test_client", request2)
        print(f"Created evolution session: {session2.id}")
        
        result2 = await manager.run_generation(session2, top_k=2)
        print(f"Evolved through {result2.generations_completed} generations")
        print(f"Total ideas: {result2.total_ideas_generated}, Time: {result2.execution_time_seconds:.1f}s")
        
        print("\nLineage sample:")
        for idea_id, parents in list(result2.lineage.items())[:3]:
            print(f"  {idea_id} <- {parents}")
        
        for i, idea in enumerate(result2.top_ideas, 1):
            print(f"\nEvolved Idea {i} (Gen {idea.generation}, Fitness: {idea.fitness:.3f}):")
            print(f"  {idea.content[:150]}...")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(test_genetic_system())