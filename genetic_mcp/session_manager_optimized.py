"""Optimized session manager with unified optimization support.

This module extends the base SessionManager to support all optimization levels
through a simple, unified API.
"""

import asyncio
import logging
from typing import Any

from .gpu_integration import GPUAcceleratedSessionManager
from .llm_client import MultiModelClient
from .models import (
    GenerationRequest,
    GenerationResult,
    Session,
)
from .optimization_config import OptimizationConfig
from .optimization_coordinator import OptimizationCoordinator
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class OptimizedSessionManager(SessionManager):
    """Session manager with integrated optimization support.

    This class provides a unified interface to all optimization levels:
    - Basic: Standard genetic algorithm (default SessionManager behavior)
    - Enhanced: Mathematical optimizations via OptimizationCoordinator
    - GPU: Hardware acceleration via GPUAcceleratedSessionManager
    - Full: All optimizations enabled

    Example:
        # Basic usage - no optimizations
        manager = OptimizedSessionManager(llm_client)

        # Enable enhanced optimizations
        config = OptimizationConfig(level="enhanced")
        manager = OptimizedSessionManager(llm_client, optimization_config=config)

        # Enable GPU acceleration
        config = OptimizationConfig(level="gpu")
        manager = OptimizedSessionManager(llm_client, optimization_config=config)

        # Enable everything
        config = OptimizationConfig(level="full")
        manager = OptimizedSessionManager(llm_client, optimization_config=config)
    """

    def __init__(
        self,
        llm_client: MultiModelClient,
        optimization_config: OptimizationConfig | None = None,
        max_sessions_per_client: int = 10,
        session_timeout_minutes: int = 60
    ):
        """Initialize optimized session manager.

        Args:
            llm_client: Multi-model LLM client
            optimization_config: Optimization configuration (defaults to basic)
            max_sessions_per_client: Maximum sessions per client
            session_timeout_minutes: Session timeout in minutes
        """
        super().__init__(
            llm_client=llm_client,
            max_sessions_per_client=max_sessions_per_client,
            session_timeout_minutes=session_timeout_minutes
        )

        # Use default config if not provided
        self.optimization_config = optimization_config or OptimizationConfig()

        # Initialize GPU session manager if needed
        self.gpu_session_manager = None
        if self.optimization_config.use_gpu:
            gpu_config = self.optimization_config.get_gpu_config()
            if gpu_config:
                try:
                    self.gpu_session_manager = GPUAcceleratedSessionManager(
                        gpu_config=gpu_config,
                        enable_gpu=True
                    )
                    logger.info("GPU acceleration initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU acceleration: {e}")
                    self.optimization_config.use_gpu = False

        # Track optimization usage per session
        self.session_optimizations: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the optimized session manager."""
        await super().start()

        # Warm up GPU if enabled
        if self.gpu_session_manager:
            try:
                await self.gpu_session_manager.warm_up_gpu()
            except Exception as e:
                logger.warning(f"GPU warmup failed: {e}")

    async def stop(self) -> None:
        """Stop the optimized session manager."""
        # Clear GPU caches if used
        if self.gpu_session_manager:
            self.gpu_session_manager.clear_gpu_caches()

        await super().stop()

    async def create_session(
        self,
        client_id: str,
        request: GenerationRequest,
        optimization_override: OptimizationConfig | None = None
    ) -> Session:
        """Create a new generation session with optimization support.

        Args:
            client_id: Client identifier
            request: Generation request
            optimization_override: Override global optimization config for this session

        Returns:
            Created session
        """
        # Create base session
        session = await super().create_session(client_id, request)

        # Determine optimization config for this session
        session_config = optimization_override or self.optimization_config

        # Store optimization info
        self.session_optimizations[session.id] = {
            "config": session_config,
            "coordinator": None,
            "gpu_enabled": False,
            "metrics": {}
        }

        # Initialize optimization coordinator if needed
        if session_config.should_use_optimization_coordinator():
            coordinator = OptimizationCoordinator(
                parameters=session.parameters,
                fitness_weights=session.fitness_weights,
                llm_client=self.llm_client if session_config.use_llm_operators else None,
                config=session_config.get_coordinator_config()
            )
            self.session_optimizations[session.id]["coordinator"] = coordinator
            logger.info(f"Optimization coordinator initialized for session {session.id}")

        # Mark if GPU is enabled for this session
        if session_config.use_gpu and self.gpu_session_manager:
            self.session_optimizations[session.id]["gpu_enabled"] = True
            logger.info(f"GPU acceleration enabled for session {session.id}")

        # Log optimization configuration
        logger.info(
            f"Created optimized session {session.id} with level: {session_config.level}"
        )

        return session

    async def run_generation(
        self,
        session: Session,
        top_k: int = 5
    ) -> GenerationResult:
        """Run generation with appropriate optimization level.

        Args:
            session: Session to run
            top_k: Number of top ideas to return

        Returns:
            Generation result
        """
        session_opt = self.session_optimizations.get(session.id, {})

        # Use optimization coordinator if available
        if session_opt.get("coordinator"):
            return await self._run_with_coordinator(session, top_k, session_opt)

        # Use GPU acceleration if available
        elif session_opt.get("gpu_enabled") and self.gpu_session_manager:
            return await self._run_with_gpu(session, top_k)

        # Fall back to standard generation
        else:
            return await super().run_generation(session, top_k)

    async def _run_with_coordinator(
        self,
        session: Session,
        top_k: int,
        session_opt: dict[str, Any]
    ) -> GenerationResult:
        """Run generation using optimization coordinator."""
        coordinator = session_opt["coordinator"]
        start_time = asyncio.get_event_loop().time()

        # Get initial population
        if session.ideas:
            initial_population = session.ideas[:session.parameters.population_size]
        else:
            # Generate initial population
            await self._update_progress(session, "Generating optimized initial population...")

            if session.client_generated:
                initial_population = await self._wait_for_client_ideas(
                    session, generation=0,
                    expected_count=session.parameters.population_size
                )
            else:
                initial_population = await session._idea_generator.generate_initial_population(
                    session.prompt,
                    session.parameters.population_size
                )
                session.ideas.extend(initial_population)

        # Get target embedding
        target_embedding = await self.llm_client.embed(session.prompt)

        # Run optimized evolution
        top_ideas, evolution_metadata = await coordinator.run_evolution(
            initial_population=initial_population,
            target_prompt=session.prompt,
            target_embedding=target_embedding,
            session=session
        )

        # Store optimization metrics
        session_opt["metrics"] = evolution_metadata

        # Build lineage from all ideas
        lineage = {}
        for idea in session.ideas:
            if idea.parent_ids:
                lineage[idea.id] = idea.parent_ids

        # Create result
        execution_time = asyncio.get_event_loop().time() - start_time

        result = GenerationResult(
            session_id=session.id,
            top_ideas=top_ideas[:top_k],
            total_ideas_generated=len(session.ideas),
            generations_completed=session.current_generation,
            lineage=lineage,
            execution_time_seconds=execution_time,
            metadata={
                "optimization_level": "enhanced",
                "evolution_metadata": evolution_metadata
            }
        )

        session.status = "completed"
        return result

    async def _run_with_gpu(
        self,
        session: Session,
        top_k: int
    ) -> GenerationResult:
        """Run generation using GPU acceleration."""
        start_time = asyncio.get_event_loop().time()

        # Get initial population
        if not session.ideas:
            await self._update_progress(session, "Generating initial population...")

            if session.client_generated:
                initial_ideas = await self._wait_for_client_ideas(
                    session, generation=0,
                    expected_count=session.parameters.population_size
                )
            else:
                initial_ideas = await session._idea_generator.generate_initial_population(
                    session.prompt,
                    session.parameters.population_size
                )
            session.ideas.extend(initial_ideas)

        # Run GPU-accelerated evolution
        current_population = session.ideas[-session.parameters.population_size:]

        for gen in range(session.parameters.generations):
            session.current_generation = gen

            new_population, progress = await self.gpu_session_manager.process_generation_gpu(
                session,
                current_population,
                session.prompt
            )

            if new_population:
                current_population = new_population

            # Update progress
            await self._update_progress(
                session,
                f"GPU Generation {gen + 1}/{session.parameters.generations}: "
                f"Best fitness = {progress.best_fitness:.3f}"
            )

        # Get diverse top ideas using GPU
        top_ideas = await self.gpu_session_manager.find_optimal_population_subset(
            session, k=top_k, diversity_weight=0.5
        )

        # Build lineage
        lineage = {}
        for idea in session.ideas:
            if idea.parent_ids:
                lineage[idea.id] = idea.parent_ids

        # Get GPU stats
        gpu_stats = self.gpu_session_manager.get_gpu_memory_stats()

        # Create result
        execution_time = asyncio.get_event_loop().time() - start_time

        result = GenerationResult(
            session_id=session.id,
            top_ideas=top_ideas,
            total_ideas_generated=len(session.ideas),
            generations_completed=session.current_generation + 1,
            lineage=lineage,
            execution_time_seconds=execution_time,
            metadata={
                "optimization_level": "gpu",
                "gpu_stats": gpu_stats
            }
        )

        session.status = "completed"
        return result

    async def get_optimization_report(self, session_id: str) -> dict[str, Any]:
        """Get detailed optimization report for a session.

        Args:
            session_id: Session identifier

        Returns:
            Optimization report with metrics and recommendations
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session_opt = self.session_optimizations.get(session_id, {})
        config = session_opt.get("config", OptimizationConfig())

        report = {
            "session_id": session_id,
            "optimization_level": config.level,
            "configuration": config.to_dict(),
            "metrics": session_opt.get("metrics", {}),
            "performance": {
                "total_ideas": len(session.ideas),
                "generations_completed": session.current_generation,
                "gpu_enabled": session_opt.get("gpu_enabled", False)
            }
        }

        # Add coordinator report if available
        if session_opt.get("coordinator"):
            coordinator_report = session_opt["coordinator"].get_optimization_report()
            report["coordinator_report"] = coordinator_report

        # Add GPU stats if available
        if session_opt.get("gpu_enabled") and self.gpu_session_manager:
            report["gpu_stats"] = self.gpu_session_manager.get_gpu_memory_stats()

        # Add recommendations
        recommendations = config.get_recommended_settings(session.parameters.population_size)
        if recommendations:
            report["recommendations"] = recommendations

        return report

    def get_global_stats(self) -> dict[str, Any]:
        """Get global optimization statistics across all sessions."""
        stats = {
            "total_sessions": len(self.sessions),
            "optimization_usage": {
                "basic": 0,
                "enhanced": 0,
                "gpu": 0,
                "full": 0
            },
            "gpu_available": self.gpu_session_manager is not None
        }

        for session_opt in self.session_optimizations.values():
            config = session_opt.get("config", OptimizationConfig())
            stats["optimization_usage"][config.level] += 1

        if self.gpu_session_manager:
            stats["gpu_memory"] = self.gpu_session_manager.get_gpu_memory_stats()

        return stats
