"""Main MCP server implementation."""

import asyncio
import logging
import os
import sys
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from .llm_client import (
    AnthropicClient,
    MultiModelClient,
    OpenAIClient,
)
from .models import (
    EvolutionMode,
    FitnessWeights,
    GenerationRequest,
)
from .session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv("GENETIC_MCP_DEBUG") else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("genetic-mcp")


# Tool schemas
class CreateSessionSchema(BaseModel):
    """Schema for create_session tool."""
    prompt: str = Field(description="The prompt to generate ideas for")
    mode: str = Field(
        default="single_pass",
        description="Evolution mode: 'single_pass' (rank top-K) or 'iterative' (genetic algorithm)"
    )
    population_size: int = Field(default=10, ge=2, description="Number of ideas per generation")
    top_k: int = Field(default=5, ge=1, description="Number of top ideas to return")
    generations: int = Field(default=5, ge=1, description="Number of generations (for iterative mode)")
    fitness_weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for fitness calculation: relevance, novelty, feasibility (must sum to 1.0)"
    )
    models: list[str] | None = Field(
        default=None,
        description="List of LLM models to use (e.g., ['openai:gpt-4', 'anthropic:claude-3'])"
    )


class RunGenerationSchema(BaseModel):
    """Schema for run_generation tool."""
    session_id: str = Field(description="The session ID to run generation for")
    top_k: int = Field(default=5, ge=1, description="Number of top ideas to return")


class GetProgressSchema(BaseModel):
    """Schema for get_progress tool."""
    session_id: str = Field(description="The session ID to get progress for")


class GetSessionSchema(BaseModel):
    """Schema for get_session tool."""
    session_id: str = Field(description="The session ID to retrieve")


class SetFitnessWeightsSchema(BaseModel):
    """Schema for set_fitness_weights tool."""
    session_id: str = Field(description="The session ID to update")
    relevance: float = Field(ge=0, le=1, description="Weight for relevance score")
    novelty: float = Field(ge=0, le=1, description="Weight for novelty score")
    feasibility: float = Field(ge=0, le=1, description="Weight for feasibility score")


# Global session manager
session_manager: SessionManager | None = None


def initialize_llm_client() -> MultiModelClient:
    """Initialize the multi-model LLM client."""
    client = MultiModelClient()

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not openai_key and not anthropic_key and not openrouter_key:
        raise ValueError("At least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY must be set")

    # Add available clients
    if openai_key:
        client.add_client("openai", OpenAIClient(openai_key), is_default=True)
        client.add_client("gpt-4", OpenAIClient(openai_key, "gpt-4-turbo-preview"))
        client.add_client("gpt-3.5", OpenAIClient(openai_key, "gpt-3.5-turbo"))

    if anthropic_key:
        anthropic_client = AnthropicClient(anthropic_key)
        if openai_key:
            anthropic_client.set_embedding_fallback(openai_key)
        client.add_client("anthropic", anthropic_client, is_default=not openai_key)
        client.add_client("claude-3", anthropic_client)

    if openrouter_key:
        from .llm_client import OpenRouterClient
        openrouter_client = OpenRouterClient(openrouter_key)
        if openai_key:
            openrouter_client.set_embedding_fallback(openai_key)
        client.add_client("openrouter", openrouter_client, is_default=(not openai_key and not anthropic_key))
        client.add_client("llama-3.2", openrouter_client)

    return client


@mcp.tool()
async def create_session(
    prompt: str,
    mode: str = "single_pass",
    population_size: int = 10,
    top_k: int = 5,
    generations: int = 5,
    fitness_weights: dict[str, float] | None = None,
    models: list[str] | None = None
) -> dict[str, Any]:
    """Create a new idea generation session.

    Args:
        prompt: The prompt to generate ideas for
        mode: Evolution mode - 'single_pass' (rank top-K) or 'iterative' (genetic algorithm)
        population_size: Number of ideas per generation
        top_k: Number of top ideas to return
        generations: Number of generations (for iterative mode)
        fitness_weights: Weights for fitness calculation (relevance, novelty, feasibility)
        models: List of LLM models to use

    Returns:
        Session information including session_id
    """
    global session_manager

    try:
        # Parse fitness weights
        weights = None
        if fitness_weights:
            weights = FitnessWeights(**fitness_weights)

        # Create request
        request = GenerationRequest(
            prompt=prompt,
            mode=EvolutionMode(mode),
            population_size=population_size,
            top_k=top_k,
            generations=generations,
            fitness_weights=weights,
            models=models
        )

        # Create session
        client_id = "default"  # In real implementation, extract from context
        session = await session_manager.create_session(client_id, request)

        return {
            "session_id": session.id,
            "status": session.status,
            "mode": session.mode,
            "parameters": {
                "population_size": session.parameters.population_size,
                "generations": session.parameters.generations,
                "mutation_rate": session.parameters.mutation_rate,
                "crossover_rate": session.parameters.crossover_rate,
                "elitism_count": session.parameters.elitism_count
            },
            "fitness_weights": {
                "relevance": session.fitness_weights.relevance,
                "novelty": session.fitness_weights.novelty,
                "feasibility": session.fitness_weights.feasibility
            }
        }

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise


@mcp.tool()
async def run_generation(session_id: str, top_k: int = 5) -> dict[str, Any]:
    """Run the generation process for a session.

    Args:
        session_id: The session ID to run generation for
        top_k: Number of top ideas to return

    Returns:
        Generation results including top ideas and lineage
    """
    global session_manager

    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != "active":
            raise ValueError(f"Session {session_id} is not active (status: {session.status})")

        # Run generation
        result = await session_manager.run_generation(session, top_k)

        # Format response
        return {
            "session_id": result.session_id,
            "top_ideas": [
                {
                    "id": idea.id,
                    "content": idea.content,
                    "generation": idea.generation,
                    "fitness": idea.fitness,
                    "scores": idea.scores,
                    "parent_ids": idea.parent_ids
                }
                for idea in result.top_ideas
            ],
            "total_ideas_generated": result.total_ideas_generated,
            "generations_completed": result.generations_completed,
            "lineage": result.lineage,
            "execution_time_seconds": result.execution_time_seconds
        }

    except Exception as e:
        logger.error(f"Error running generation: {e}")
        raise


@mcp.tool()
async def get_progress(session_id: str) -> dict[str, Any]:
    """Get progress information for a running session.

    Args:
        session_id: The session ID to get progress for

    Returns:
        Progress information including current generation and active workers
    """
    global session_manager

    try:
        progress = await session_manager.get_progress(session_id)
        if not progress:
            raise ValueError(f"Session {session_id} not found")

        return {
            "session_id": progress.session_id,
            "current_generation": progress.current_generation,
            "total_generations": progress.total_generations,
            "ideas_generated": progress.ideas_generated,
            "active_workers": progress.active_workers,
            "best_fitness": progress.best_fitness,
            "status": progress.status,
            "message": progress.message
        }

    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise


@mcp.tool()
async def get_session(session_id: str) -> dict[str, Any]:
    """Get detailed information about a session.

    Args:
        session_id: The session ID to retrieve

    Returns:
        Detailed session information including all ideas
    """
    global session_manager

    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        return {
            "session_id": session.id,
            "client_id": session.client_id,
            "prompt": session.prompt,
            "mode": session.mode,
            "status": session.status,
            "current_generation": session.current_generation,
            "parameters": {
                "population_size": session.parameters.population_size,
                "generations": session.parameters.generations,
                "mutation_rate": session.parameters.mutation_rate,
                "crossover_rate": session.parameters.crossover_rate,
                "elitism_count": session.parameters.elitism_count
            },
            "fitness_weights": {
                "relevance": session.fitness_weights.relevance,
                "novelty": session.fitness_weights.novelty,
                "feasibility": session.fitness_weights.feasibility
            },
            "ideas": [
                {
                    "id": idea.id,
                    "content": idea.content,
                    "generation": idea.generation,
                    "fitness": idea.fitness,
                    "scores": idea.scores,
                    "parent_ids": idea.parent_ids
                }
                for idea in session.ideas
            ],
            "worker_stats": session._worker_pool.get_worker_stats() if hasattr(session, '_worker_pool') else {}
        }

    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise


@mcp.tool()
async def set_fitness_weights(
    session_id: str,
    relevance: float,
    novelty: float,
    feasibility: float
) -> dict[str, str]:
    """Update fitness weights for a session.

    Args:
        session_id: The session ID to update
        relevance: Weight for relevance score (0-1)
        novelty: Weight for novelty score (0-1)
        feasibility: Weight for feasibility score (0-1)

    Returns:
        Success message
    """
    global session_manager

    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Update weights
        session.fitness_weights = FitnessWeights(
            relevance=relevance,
            novelty=novelty,
            feasibility=feasibility
        )

        # Update fitness evaluator if it exists
        if hasattr(session, '_fitness_evaluator'):
            session._fitness_evaluator.weights = session.fitness_weights

        return {"message": "Fitness weights updated successfully"}

    except Exception as e:
        logger.error(f"Error updating fitness weights: {e}")
        raise


async def initialize_server():
    """Initialize the server components."""
    global session_manager

    # Initialize LLM client
    llm_client = initialize_llm_client()

    # Initialize session manager
    session_manager = SessionManager(llm_client)
    await session_manager.start()

    logger.info("Genetic MCP server initialized")


async def shutdown_server():
    """Shutdown the server components."""
    global session_manager

    if session_manager:
        await session_manager.stop()

    logger.info("Genetic MCP server shutdown")


def main():
    """Main entry point."""
    # Determine transport mode
    transport = os.getenv("GENETIC_MCP_TRANSPORT", "stdio").lower()

    if transport == "stdio":
        # Initialize server synchronously before running
        async def init():
            await initialize_server()

        asyncio.run(init())

        # Set up shutdown handler
        import atexit
        import signal
        
        def sync_shutdown():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(shutdown_server())
                loop.close()
            except Exception:
                pass  # Ignore errors during shutdown
        
        atexit.register(sync_shutdown)

        # Run in stdio mode - FastMCP handles the async context
        mcp.run(transport="stdio")

    elif transport == "http":
        # Run in HTTP mode with SSE
        from .transport import create_http_app

        app = create_http_app(mcp, initialize_server, shutdown_server)

        import uvicorn
        host = os.getenv("GENETIC_MCP_HOST", "0.0.0.0")
        port = int(os.getenv("GENETIC_MCP_PORT", "3000"))

        uvicorn.run(app, host=host, port=port)

    else:
        logger.error(f"Unknown transport mode: {transport}")
        sys.exit(1)


if __name__ == "__main__":
    main()
