#!/usr/bin/env python3
"""Simple integration test for Genetic MCP server."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_mcp.llm_client import MultiModelClient, OpenRouterClient
from genetic_mcp.models import GenerationRequest, EvolutionMode
from genetic_mcp.session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_openrouter_client():
    """Test OpenRouter client directly."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return False
    
    client = OpenRouterClient(api_key, "meta-llama/llama-3.2-3b-instruct")
    
    try:
        # Test generation
        response = await client.generate(
            "Generate a creative idea for a sustainable urban transportation system",
            system_prompt="You are a creative innovation assistant.",
            temperature=0.8
        )
        logger.info(f"OpenRouter response: {response[:200]}...")
        
        # Test embedding (will use dummy)
        embedding = await client.embed("test text")
        logger.info(f"Embedding dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        logger.error(f"OpenRouter test failed: {e}")
        return False


async def test_session_manager():
    """Test the session manager with a simple generation."""
    # Initialize LLM client
    client = MultiModelClient()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return False
    
    openrouter_client = OpenRouterClient(api_key)
    client.add_client("openrouter", openrouter_client, is_default=True)
    
    # Create session manager
    manager = SessionManager(client)
    await manager.start()
    
    try:
        # Create a generation request
        request = GenerationRequest(
            prompt="Generate innovative ideas for reducing plastic waste in oceans",
            mode=EvolutionMode.SINGLE_PASS,
            population_size=5,
            top_k=3
        )
        
        # Create session
        session = await manager.create_session("test_client", request)
        logger.info(f"Created session: {session.id}")
        
        # Run generation
        result = await manager.run_generation(session, top_k=3)
        
        # Display results
        logger.info(f"\nGeneration completed in {result.execution_time_seconds:.2f} seconds")
        logger.info(f"Total ideas generated: {result.total_ideas_generated}")
        logger.info(f"\nTop {len(result.top_ideas)} ideas:")
        
        for i, idea in enumerate(result.top_ideas, 1):
            logger.info(f"\n--- Idea {i} ---")
            logger.info(f"Fitness: {idea.fitness:.3f}")
            logger.info(f"Scores: Relevance={idea.scores.get('relevance', 0):.3f}, "
                       f"Novelty={idea.scores.get('novelty', 0):.3f}, "
                       f"Feasibility={idea.scores.get('feasibility', 0):.3f}")
            logger.info(f"Content: {idea.content[:300]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Session manager test failed: {e}", exc_info=True)
        return False
    finally:
        await manager.stop()


async def test_genetic_evolution():
    """Test genetic algorithm evolution."""
    # Initialize LLM client
    client = MultiModelClient()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return False
    
    openrouter_client = OpenRouterClient(api_key)
    client.add_client("openrouter", openrouter_client, is_default=True)
    
    # Create session manager
    manager = SessionManager(client)
    await manager.start()
    
    try:
        # Create a generation request with iterative evolution
        request = GenerationRequest(
            prompt="Design a futuristic educational system that combines AI and human creativity",
            mode=EvolutionMode.ITERATIVE,
            population_size=6,
            top_k=3,
            generations=3
        )
        
        # Create session
        session = await manager.create_session("test_client", request)
        logger.info(f"Created session for genetic evolution: {session.id}")
        
        # Run generation
        result = await manager.run_generation(session, top_k=3)
        
        # Display results
        logger.info(f"\nGenetic evolution completed in {result.execution_time_seconds:.2f} seconds")
        logger.info(f"Generations completed: {result.generations_completed}")
        logger.info(f"Total ideas generated: {result.total_ideas_generated}")
        
        # Show lineage
        logger.info("\nIdea lineage:")
        for idea_id, parent_ids in list(result.lineage.items())[:5]:
            logger.info(f"  {idea_id} <- {parent_ids}")
        
        logger.info(f"\nTop {len(result.top_ideas)} evolved ideas:")
        for i, idea in enumerate(result.top_ideas, 1):
            logger.info(f"\n--- Evolved Idea {i} ---")
            logger.info(f"Generation: {idea.generation}")
            logger.info(f"Fitness: {idea.fitness:.3f}")
            logger.info(f"Parent IDs: {idea.parent_ids}")
            logger.info(f"Content: {idea.content[:400]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Genetic evolution test failed: {e}", exc_info=True)
        return False
    finally:
        await manager.stop()


async def main():
    """Run all tests."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("Starting Genetic MCP integration tests...\n")
    
    # Test 1: OpenRouter client
    logger.info("=== Test 1: OpenRouter Client ===")
    if await test_openrouter_client():
        logger.info("✓ OpenRouter client test passed\n")
    else:
        logger.error("✗ OpenRouter client test failed\n")
    
    # Test 2: Session Manager with single pass
    logger.info("=== Test 2: Session Manager (Single Pass) ===")
    if await test_session_manager():
        logger.info("✓ Session manager test passed\n")
    else:
        logger.error("✗ Session manager test failed\n")
    
    # Test 3: Genetic Evolution
    logger.info("=== Test 3: Genetic Evolution ===")
    if await test_genetic_evolution():
        logger.info("✓ Genetic evolution test passed\n")
    else:
        logger.error("✗ Genetic evolution test failed\n")
    
    logger.info("Integration tests completed!")


if __name__ == "__main__":
    asyncio.run(main())