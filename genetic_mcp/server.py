"""Main MCP server implementation."""

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

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
from .optimization_config import OptimizationConfig
from .session_manager import SessionManager
from .session_manager_optimized import OptimizedSessionManager

# Load environment variables
load_dotenv()

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
    client_generated: bool = Field(
        default=False,
        description="Whether the client generates ideas instead of using LLM workers"
    )
    optimization_level: str | None = Field(
        default=None,
        description="Optimization level: 'basic', 'enhanced', 'gpu', or 'full'. If not specified, uses server default."
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


class InjectIdeasSchema(BaseModel):
    """Schema for inject_ideas tool."""
    session_id: str = Field(description="The session ID to inject ideas into")
    ideas: list[str] = Field(description="List of idea contents to inject")
    generation: int = Field(default=0, ge=0, description="Generation number for these ideas")


class GetOptimizationStatsSchema(BaseModel):
    """Schema for get_optimization_stats tool."""
    pass  # No parameters needed


class GetOptimizationReportSchema(BaseModel):
    """Schema for get_optimization_report tool."""
    session_id: str = Field(description="The session ID to get optimization report for")


# Global session manager
session_manager: SessionManager | None = None


def initialize_llm_client() -> MultiModelClient:
    """Initialize the multi-model LLM client."""
    client = MultiModelClient()

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    # Get model configurations from environment
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct")

    if not openai_key and not anthropic_key and not openrouter_key:
        raise ValueError("At least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY must be set")

    # Add available clients
    if openai_key:
        client.add_client("openai", OpenAIClient(openai_key, openai_model), is_default=True)
        client.add_client("gpt-4", OpenAIClient(openai_key, "gpt-4-turbo-preview"))
        client.add_client("gpt-3.5", OpenAIClient(openai_key, "gpt-3.5-turbo"))

    if anthropic_key:
        anthropic_client = AnthropicClient(anthropic_key, anthropic_model)
        if openai_key:
            anthropic_client.set_embedding_fallback(openai_key)
        client.add_client("anthropic", anthropic_client, is_default=not openai_key)
        client.add_client("claude-3", anthropic_client)

    if openrouter_key:
        from .llm_client import OpenRouterClient
        openrouter_client = OpenRouterClient(openrouter_key, openrouter_model)
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
    models: list[str] | None = None,
    client_generated: bool = False,
    optimization_level: str | None = None
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
        client_generated: Whether the client generates ideas instead of using LLM workers

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
            models=models,
            client_generated=client_generated
        )

        # Create session
        client_id = "default"  # In real implementation, extract from context

        # Handle optimization if supported
        if hasattr(session_manager, 'create_session') and callable(getattr(session_manager.create_session, '__func__', None)):
            # Check if this is an OptimizedSessionManager
            if optimization_level and hasattr(session_manager, 'optimization_config'):
                # Create session-specific optimization config
                from .optimization_config import OptimizationConfig
                session_config = OptimizationConfig(level=optimization_level)
                session = await session_manager.create_session(client_id, request, optimization_override=session_config)
            else:
                session = await session_manager.create_session(client_id, request)
        else:
            session = await session_manager.create_session(client_id, request)

        return {
            "session_id": session.id,
            "status": session.status,
            "mode": session.mode,
            "client_generated": session.client_generated,
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
async def get_session(
    session_id: str,
    include_ideas: bool = True,
    ideas_limit: int | None = 100,
    ideas_offset: int = 0,
    generation_filter: int | None = None
) -> dict[str, Any]:
    """Get detailed information about a session with pagination support.

    Args:
        session_id: The session ID to retrieve
        include_ideas: Whether to include ideas in the response (default: True)
        ideas_limit: Maximum number of ideas to return (default: 100, max: 1000)
        ideas_offset: Offset for pagination (default: 0)
        generation_filter: Only return ideas from a specific generation (optional)

    Returns:
        Detailed session information with paginated ideas
    """
    global session_manager

    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Validate and set limits
        if ideas_limit is None:
            ideas_limit = 100
        ideas_limit = min(ideas_limit, 1000)  # Cap at 1000 ideas

        result = {
            "session_id": session.id,
            "client_id": session.client_id,
            "prompt": session.prompt,
            "mode": session.mode,
            "status": session.status,
            "client_generated": session.client_generated,
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
            "total_ideas": len(session.ideas),
            "ideas_per_generation": session.ideas_per_generation_received if session.client_generated else {},
            "worker_stats": session._worker_pool.get_worker_stats() if hasattr(session, '_worker_pool') and not session.client_generated else {}
        }

        # Add ideas with pagination if requested
        if include_ideas:
            # Filter by generation if specified
            ideas = session.ideas
            if generation_filter is not None:
                ideas = [idea for idea in ideas if idea.generation == generation_filter]

            # Apply pagination
            total_filtered = len(ideas)
            ideas_page = ideas[ideas_offset:ideas_offset + ideas_limit]

            result["ideas"] = [
                {
                    "id": idea.id,
                    "content": idea.content,
                    "generation": idea.generation,
                    "fitness": idea.fitness,
                    "scores": idea.scores,
                    "parent_ids": idea.parent_ids
                }
                for idea in ideas_page
            ]
            result["ideas_returned"] = len(ideas_page)
            result["ideas_total_filtered"] = total_filtered
            result["ideas_offset"] = ideas_offset
            result["ideas_limit"] = ideas_limit
            result["has_more_ideas"] = (ideas_offset + ideas_limit) < total_filtered

        return result

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


@mcp.tool()
async def get_optimization_stats() -> dict[str, Any]:
    """Get optimization statistics and capabilities.

    Returns:
        Dictionary containing optimization capabilities and current usage statistics
    """
    global session_manager

    try:
        # Check if using optimized session manager
        if hasattr(session_manager, 'get_global_stats'):
            stats = session_manager.get_global_stats()
        else:
            stats = {
                "optimization_enabled": False,
                "message": "Standard session manager in use. Set GENETIC_MCP_OPTIMIZATION_ENABLED=true to enable optimizations."
            }

        # Add server configuration info
        stats["configuration"] = {
            "optimization_enabled": os.getenv("GENETIC_MCP_OPTIMIZATION_ENABLED", "false").lower() == "true",
            "optimization_level": os.getenv("GENETIC_MCP_OPTIMIZATION_LEVEL", "basic"),
            "gpu_requested": os.getenv("GENETIC_MCP_USE_GPU", "false").lower() == "true"
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting optimization stats: {e}")
        raise


@mcp.tool()
async def get_optimization_report(session_id: str) -> dict[str, Any]:
    """Get detailed optimization report for a specific session.

    Args:
        session_id: The session ID to get report for

    Returns:
        Detailed optimization report including metrics, configuration, and recommendations
    """
    global session_manager

    try:
        # Check if using optimized session manager
        if hasattr(session_manager, 'get_optimization_report'):
            report = await session_manager.get_optimization_report(session_id)
            return report
        else:
            return {
                "session_id": session_id,
                "optimization_enabled": False,
                "message": "Optimization reports are only available when using the optimized session manager."
            }

    except Exception as e:
        logger.error(f"Error getting optimization report: {e}")
        raise


@mcp.tool()
async def inject_ideas(
    session_id: str,
    ideas: list[str],
    generation: int = 0
) -> dict[str, Any]:
    """Inject client-generated ideas into a session.

    Args:
        session_id: The session ID to inject ideas into
        ideas: List of idea contents to inject
        generation: Generation number for these ideas (default: 0)

    Returns:
        Information about injected ideas
    """
    global session_manager

    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.client_generated:
            raise ValueError(f"Session {session_id} is not configured for client-generated ideas")

        if session.status != "active":
            raise ValueError(f"Session {session_id} is not active (status: {session.status})")

        # Inject ideas into the session
        injected_ideas = await session_manager.inject_ideas(session, ideas, generation)

        return {
            "session_id": session_id,
            "injected_count": len(injected_ideas),
            "generation": generation,
            "total_ideas": len(session.ideas),
            "ideas_per_generation": session.ideas_per_generation_received,
            "message": f"Successfully injected {len(injected_ideas)} ideas into generation {generation}"
        }

    except Exception as e:
        logger.error(f"Error injecting ideas: {e}")
        raise


async def initialize_server():
    """Initialize the server components."""
    global session_manager

    # Initialize LLM client
    llm_client = initialize_llm_client()

    # Check if optimizations are requested
    optimization_enabled = os.getenv("GENETIC_MCP_OPTIMIZATION_ENABLED", "false").lower() == "true"

    if optimization_enabled:
        # Create optimization config from environment
        optimization_config = OptimizationConfig.from_env()

        # Initialize optimized session manager
        session_manager = OptimizedSessionManager(
            llm_client=llm_client,
            optimization_config=optimization_config
        )
        logger.info(f"Genetic MCP server initialized with optimization level: {optimization_config.level}")
    else:
        # Initialize standard session manager
        session_manager = SessionManager(llm_client)
        logger.info("Genetic MCP server initialized (standard mode)")

    await session_manager.start()


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
