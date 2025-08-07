"""Session management for genetic MCP server."""

import asyncio
import time
import uuid
from asyncio import TimeoutError
from datetime import datetime, timedelta

from .adaptive_population import AdaptivePopulationManager, PopulationConfig
from .diversity_manager import DiversityManager
from .fitness import FitnessEvaluator
from .genetic_algorithm import GeneticAlgorithm
from .llm_client import MultiModelClient
from .logging_config import log_error, log_operation, log_performance, setup_logging
from .memory_system import get_memory_system
from .models import (
    EvolutionMode,
    FitnessWeights,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    GeneticParameters,
    Idea,
    Session,
)
from .persistence_manager import PersistenceManager
from .worker_pool import IdeaGenerator, WorkerPool

logger = setup_logging(component="session_manager")


class SessionManager:
    """Manages generation sessions."""

    def __init__(self, llm_client: MultiModelClient,
                 max_sessions_per_client: int = 10,
                 session_timeout_minutes: int = 60,
                 persistence_db_path: str = "genetic_mcp_sessions.db",
                 enable_auto_save: bool = True):
        self.llm_client = llm_client
        self.max_sessions_per_client = max_sessions_per_client
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions: dict[str, Session] = {}
        self.client_sessions: dict[str, list[str]] = {}
        self._cleanup_task: asyncio.Task | None = None

        # Persistence configuration
        self.persistence_manager = PersistenceManager(persistence_db_path)
        self.enable_auto_save = enable_auto_save
        self._auto_save_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the session manager."""
        # Initialize persistence manager
        await self.persistence_manager.initialize()

        # Start cleanup and auto-save tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self.enable_auto_save:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        logger.info("Session manager started with persistence enabled")

    async def stop(self) -> None:
        """Stop the session manager."""
        # Cancel background tasks
        tasks_to_cancel = []
        if self._cleanup_task:
            tasks_to_cancel.append(self._cleanup_task)
        if self._auto_save_task:
            tasks_to_cancel.append(self._auto_save_task)

        for task in tasks_to_cancel:
            task.cancel()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Save all active sessions before shutdown
        if self.enable_auto_save:
            await self._save_all_sessions()

        # Stop all active sessions
        for session in self.sessions.values():
            if hasattr(session, '_worker_pool') and session._worker_pool is not None:
                await session._worker_pool.stop()

        logger.info("Session manager stopped")

    async def create_session(self, client_id: str, request: GenerationRequest) -> Session:
        """Create a new generation session."""
        start_time = time.time()

        log_operation(logger, "CREATE_SESSION",
                      client_id=client_id,
                      mode=request.mode,
                      population_size=request.population_size,
                      client_generated=request.client_generated)

        # Check client session limit
        client_sessions = self.client_sessions.get(client_id, [])
        if len(client_sessions) >= self.max_sessions_per_client:
            raise ValueError(f"Client {client_id} has reached maximum session limit")

        # Get memory system for parameter optimization
        memory_system = get_memory_system()
        parameter_recommendation = None

        # Get parameter recommendations from memory system if enabled
        if request.use_memory_system and memory_system.enable_learning:
            try:
                parameter_recommendation = await memory_system.get_parameter_recommendation(request.prompt)
                if parameter_recommendation:
                    logger.info(f"Memory system provided parameter recommendation with confidence {parameter_recommendation.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Failed to get memory recommendation: {e}")

        # Create session ID first
        session_id = str(uuid.uuid4())

        # Use parameters from request, recommendation, or defaults (in that order)
        if request.parameters:
            parameters = request.parameters
        elif parameter_recommendation and parameter_recommendation.confidence > 0.7:
            parameters = parameter_recommendation.parameters
            logger.info(f"Using memory-recommended parameters for session {session_id}")
        else:
            parameters = GeneticParameters(
                population_size=request.population_size,
                generations=request.generations
            )

        # Use fitness weights from request, recommendation, or defaults
        if request.fitness_weights:
            fitness_weights = request.fitness_weights
        elif parameter_recommendation and parameter_recommendation.confidence > 0.7:
            fitness_weights = parameter_recommendation.fitness_weights
            logger.info(f"Using memory-recommended fitness weights for session {session_id}")
        else:
            fitness_weights = FitnessWeights()
        session = Session(
            id=session_id,
            client_id=client_id,
            prompt=request.prompt,
            mode=request.mode,
            client_generated=request.client_generated,
            parameters=parameters,
            fitness_weights=fitness_weights,
            adaptive_population_enabled=request.adaptive_population,
            adaptive_population_config=request.adaptive_population_config or {},
            memory_enabled=request.use_memory_system,
            parameter_recommendation=parameter_recommendation.model_dump() if parameter_recommendation else {}
        )

        # Initialize session components
        session._fitness_evaluator = FitnessEvaluator(session.fitness_weights)
        session._genetic_algorithm = GeneticAlgorithm(session.parameters)

        # Initialize diversity manager (needed for adaptive population)
        session._diversity_manager = DiversityManager()

        # Initialize adaptive population manager if enabled
        if session.adaptive_population_enabled:
            population_config = PopulationConfig(**session.adaptive_population_config)
            session._adaptive_population_manager = AdaptivePopulationManager(population_config)
            logger.info(f"Initialized adaptive population manager for session {session_id}")

        # Only initialize worker pool if not client-generated
        if not session.client_generated:
            session._worker_pool = WorkerPool(self.llm_client, max_workers=20)
            session._idea_generator = IdeaGenerator(session._worker_pool, self.llm_client)
            # Start worker pool
            await session._worker_pool.start()

        # Store session
        self.sessions[session_id] = session
        if client_id not in self.client_sessions:
            self.client_sessions[client_id] = []
        self.client_sessions[client_id].append(session_id)

        log_performance(logger, "CREATE_SESSION", time.time() - start_time,
                       session_id=session_id,
                       client_id=client_id,
                       mode=request.mode)

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        log_operation(logger, "DELETE_SESSION", session_id=session_id)

        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Attempted to delete non-existent session {session_id}")
            return

        # Stop worker pool (only if not client-generated)
        if hasattr(session, '_worker_pool') and session._worker_pool is not None:
            await session._worker_pool.stop()

        # Remove from client sessions
        if session.client_id in self.client_sessions:
            self.client_sessions[session.client_id].remove(session_id)
            if not self.client_sessions[session.client_id]:
                del self.client_sessions[session.client_id]

        # Remove session
        del self.sessions[session_id]
        logger.info(f"Successfully deleted session {session_id}")

    async def inject_ideas(self, session: Session, ideas: list[str], generation: int) -> list[Idea]:
        """Inject client-generated ideas into a session."""
        log_operation(logger, "INJECT_IDEAS",
                      session_id=session.id,
                      idea_count=len(ideas),
                      generation=generation)

        # Validate session state
        if not session.client_generated:
            raise ValueError(f"Session {session.id} is not configured for client-generated ideas")

        if session.status != "active":
            raise ValueError(f"Cannot inject ideas into session with status '{session.status}'")

        injected_ideas = []

        for i, content in enumerate(ideas):
            idea = Idea(
                id=str(uuid.uuid4()),
                content=content,
                generation=generation,
                metadata={
                    "source": "client",
                    "injection_index": i
                }
            )
            session.ideas.append(idea)
            injected_ideas.append(idea)

        # Track ideas per generation
        if generation not in session.ideas_per_generation_received:
            session.ideas_per_generation_received[generation] = 0
        session.ideas_per_generation_received[generation] += len(ideas)

        # Update session timestamp
        session.updated_at = datetime.utcnow()

        logger.info(f"Injected {len(ideas)} ideas into session {session.id} for generation {generation}")
        return injected_ideas

    async def run_generation(self, session: Session, top_k: int = 5) -> GenerationResult:
        """Run the generation process for a session."""
        start_time = datetime.utcnow()
        perf_start = time.time()

        log_operation(logger, "RUN_GENERATION",
                      session_id=session.id,
                      mode=session.mode,
                      population_size=session.parameters.population_size,
                      generations=session.parameters.generations,
                      top_k=top_k)

        try:
            # Handle initial population
            if session.client_generated:
                await self._update_progress(session, "Waiting for initial population from client...")

                # Wait for client to inject initial ideas
                initial_ideas = await self._wait_for_client_ideas(
                    session,
                    generation=0,
                    expected_count=session.parameters.population_size
                )
            else:
                await self._update_progress(session, "Generating initial population...")

                initial_ideas = await session._idea_generator.generate_initial_population(
                    session.prompt,
                    session.parameters.population_size
                )
                session.ideas.extend(initial_ideas)

            # Get embeddings for all ideas and the target prompt
            await self._update_progress(session, "Generating embeddings...")
            target_embedding = await self.llm_client.embed(session.prompt)

            for idea in initial_ideas:
                embedding = await self.llm_client.embed(idea.content)
                session._fitness_evaluator.add_embedding(idea.id, embedding)

            # Evaluate initial population
            session._fitness_evaluator.evaluate_population(initial_ideas, target_embedding)

            if session.mode == EvolutionMode.SINGLE_PASS:
                # Just return top-K
                top_ideas = session.get_top_ideas(top_k)
            else:
                # Run genetic algorithm
                current_population = initial_ideas

                for generation in range(1, session.parameters.generations):
                    session.current_generation = generation
                    await self._update_progress(
                        session,
                        f"Running generation {generation}/{session.parameters.generations}..."
                    )

                    # Calculate diversity metrics if adaptive population is enabled
                    diversity_metrics = None
                    if session.adaptive_population_enabled and hasattr(session, '_diversity_manager'):
                        # Get embeddings for diversity calculation
                        embeddings = {
                            idea.id: session._fitness_evaluator.embeddings_cache.get(idea.id)
                            for idea in current_population
                            if idea.id in session._fitness_evaluator.embeddings_cache
                        }
                        diversity_metrics = session._diversity_manager.calculate_diversity_metrics(
                            current_population, embeddings
                        )
                        logger.debug(f"Generation {generation} diversity metrics: {diversity_metrics}")

                    # Adjust population size if adaptive population is enabled
                    if session.adaptive_population_enabled:
                        recommended_size = session.get_recommended_population_size(diversity_metrics)
                        if recommended_size != session.parameters.population_size:
                            old_size = session.parameters.population_size
                            session.update_population_size_dynamically(recommended_size)
                            logger.info(f"Adaptive population: adjusted from {old_size} to {recommended_size} for generation {generation}")

                    # Get selection probabilities
                    probabilities = session._fitness_evaluator.get_selection_probabilities(
                        current_population
                    )

                    if session.client_generated:
                        # Request ideas from client for next generation
                        await self._update_progress(
                            session,
                            f"Waiting for generation {generation} ideas from client..."
                        )

                        # Wait for client to inject ideas for this generation
                        new_population = await self._wait_for_client_ideas(
                            session,
                            generation=generation,
                            expected_count=session.parameters.population_size,
                            parent_population=current_population
                        )
                    else:
                        # Create next generation
                        new_population = session._genetic_algorithm.create_next_generation(
                            current_population,
                            probabilities,
                            generation
                        )

                        # Generate content for new ideas using LLM
                        for idea in new_population:
                            if not idea.metadata.get("elite", False):
                                # Generate new content based on parents
                                parent_contents = [
                                    p.content for p in current_population
                                    if p.id in idea.parent_ids
                                ]

                                generated = await session._idea_generator.generate_from_parents(
                                    parent_contents,
                                    session.prompt,
                                    1
                                )

                                if generated:
                                    idea.content = generated[0].content

                    # Get embeddings for new ideas
                    for idea in new_population:
                        if idea.id not in session._fitness_evaluator.embeddings_cache:
                            embedding = await self.llm_client.embed(idea.content)
                            session._fitness_evaluator.add_embedding(idea.id, embedding)

                    # Evaluate new population
                    session._fitness_evaluator.evaluate_population(new_population, target_embedding)

                    # Add to session ideas
                    session.ideas.extend(new_population)
                    current_population = new_population

                    # Save checkpoint every few generations
                    if self.enable_auto_save and generation % 3 == 0:
                        try:
                            await self.persistence_manager.save_checkpoint(
                                session,
                                f"generation_{generation}",
                                {"population_size": len(current_population), "best_fitness": max(idea.fitness for idea in current_population) if current_population else 0.0}
                            )
                            logger.debug(f"Saved checkpoint for session {session.id} at generation {generation}")
                        except Exception as e:
                            logger.warning(f"Failed to save checkpoint for session {session.id}: {e}")

                # Get top ideas from final population
                top_ideas = session.get_top_ideas(top_k)

            # Build lineage
            lineage = {}
            for idea in session.ideas:
                if idea.parent_ids:
                    lineage[idea.id] = idea.parent_ids

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            session.execution_time_seconds = execution_time

            # Store session results in memory system for learning
            if session.memory_enabled:
                try:
                    memory_system = get_memory_system()
                    await memory_system.store_session_results(session)
                except Exception as e:
                    logger.warning(f"Failed to store session results in memory system: {e}")

            # Create result
            result = GenerationResult(
                session_id=session.id,
                top_ideas=top_ideas,
                total_ideas_generated=len(session.ideas),
                generations_completed=session.current_generation + 1,
                lineage=lineage,
                execution_time_seconds=execution_time
            )

            session.status = "completed"

            # Auto-save completed session
            if self.enable_auto_save:
                try:
                    await self.persistence_manager.save_session(session, "generation_completed")
                    logger.debug(f"Auto-saved completed session {session.id}")
                except Exception as e:
                    logger.warning(f"Failed to auto-save completed session {session.id}: {e}")

            log_performance(logger, "RUN_GENERATION", time.time() - perf_start,
                           session_id=session.id,
                           total_ideas=result.total_ideas_generated,
                           generations=result.generations_completed,
                           execution_time=execution_time,
                           status="completed")

            return result

        except Exception as e:
            session.status = "failed"
            log_error(logger, "RUN_GENERATION", e, session_id=session.id)
            raise

    async def get_progress(self, session_id: str) -> GenerationProgress | None:
        """Get progress for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        best_fitness = max(idea.fitness for idea in session.ideas) if session.ideas else 0.0

        return GenerationProgress(
            session_id=session.id,
            current_generation=session.current_generation,
            total_generations=session.parameters.generations,
            ideas_generated=len(session.ideas),
            active_workers=len(session.get_active_workers()),
            best_fitness=best_fitness,
            status=session.status
        )

    async def _update_progress(self, session: Session, message: str) -> None:
        """Update session progress."""
        session.updated_at = datetime.utcnow()
        logger.info(f"Session {session.id}: {message}")

    async def _wait_for_client_ideas(self, session: Session, generation: int,
                                    expected_count: int, parent_population: list[Idea] | None = None,
                                    timeout_seconds: int = 300) -> list[Idea]:
        """Wait for client to inject ideas for a specific generation."""
        start_time = datetime.utcnow()

        # Get initial count of ideas for this generation
        initial_count = session.ideas_per_generation_received.get(generation, 0)

        while True:
            # Check if we have enough ideas for this generation
            current_count = session.ideas_per_generation_received.get(generation, 0)
            if current_count - initial_count >= expected_count:
                # Get the ideas for this generation
                generation_ideas = [
                    idea for idea in session.ideas
                    if idea.generation == generation and idea.metadata.get("source") == "client"
                ][-expected_count:]  # Get the last expected_count ideas

                # If we have parent population, we might want to set parent_ids
                # This is handled by the client when creating ideas

                return generation_ideas

            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timeout waiting for client ideas for generation {generation}. "
                    f"Expected {expected_count}, received {current_count - initial_count}"
                )

            # Wait a bit before checking again
            await asyncio.sleep(0.5)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                now = datetime.utcnow()
                expired_sessions = []

                for session_id, session in self.sessions.items():
                    if (session.status == "completed" or session.status == "failed") and \
                       (now - session.updated_at > self.session_timeout):
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    await self.delete_session(session_id)
                    logger.info(f"Cleaned up expired session {session_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _auto_save_loop(self) -> None:
        """Periodically auto-save active sessions."""
        while True:
            try:
                await asyncio.sleep(180)  # Auto-save every 3 minutes
                await self._save_all_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")

    async def _save_all_sessions(self) -> None:
        """Save all active sessions to database."""
        if not self.sessions:
            return

        saved_count = 0
        for session_id, session in self.sessions.items():
            try:
                # Only save active or recently completed sessions
                if session.status in ["active", "completed"]:
                    await self.persistence_manager.save_session(session)
                    saved_count += 1
            except Exception as e:
                logger.error(f"Failed to auto-save session {session_id}: {e}")

        if saved_count > 0:
            logger.debug(f"Auto-saved {saved_count} sessions")

    # Persistence methods
    async def save_session_to_db(self, session_id: str, checkpoint_name: str | None = None) -> bool:
        """Save a session to the database."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        try:
            await self.persistence_manager.save_session(session, checkpoint_name)
            logger.info(f"Successfully saved session {session_id} to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    async def load_session_from_db(self, session_id: str) -> Session | None:
        """Load a session from the database."""
        try:
            session = await self.persistence_manager.load_session(session_id)
            if session:
                # Reconstruct session components
                await self._reconstruct_session_components(session)

                # Add to active sessions
                self.sessions[session_id] = session
                if session.client_id not in self.client_sessions:
                    self.client_sessions[session.client_id] = []
                if session_id not in self.client_sessions[session.client_id]:
                    self.client_sessions[session.client_id].append(session_id)

                logger.info(f"Successfully loaded session {session_id} from database")
                return session
            else:
                logger.warning(f"Session {session_id} not found in database")
                return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def resume_session(self, session_id: str) -> bool:
        """Resume a session from the database."""
        session = await self.load_session_from_db(session_id)
        if not session:
            return False

        try:
            # Update status to active if it was completed/failed
            if session.status in ["completed", "failed", "paused"]:
                session.status = "active"
                session.updated_at = datetime.utcnow()

            # Start worker pool if not client-generated and not already started
            if not session.client_generated and not hasattr(session, '_worker_pool'):
                session._worker_pool = WorkerPool(self.llm_client, max_workers=20)
                session._idea_generator = IdeaGenerator(session._worker_pool, self.llm_client)
                await session._worker_pool.start()

            logger.info(f"Successfully resumed session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return False

    async def list_saved_sessions(self, client_id: str | None = None, limit: int = 50, offset: int = 0):
        """List saved sessions from the database."""
        try:
            return await self.persistence_manager.list_saved_sessions(client_id, limit, offset)
        except Exception as e:
            logger.error(f"Failed to list saved sessions: {e}")
            return []

    async def save_checkpoint(self, session_id: str, checkpoint_name: str, additional_data: dict | None = None) -> bool:
        """Save a checkpoint for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        try:
            await self.persistence_manager.save_checkpoint(session, checkpoint_name, additional_data)
            logger.info(f"Saved checkpoint '{checkpoint_name}' for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint for session {session_id}: {e}")
            return False

    async def _reconstruct_session_components(self, session: Session) -> None:
        """Reconstruct session components after loading from database."""
        try:
            # Initialize fitness evaluator
            session._fitness_evaluator = FitnessEvaluator(session.fitness_weights)

            # Initialize genetic algorithm
            session._genetic_algorithm = GeneticAlgorithm(session.parameters)

            # Initialize diversity manager
            session._diversity_manager = DiversityManager()

            # Initialize adaptive population manager if enabled
            if session.adaptive_population_enabled:
                population_config = PopulationConfig(**session.adaptive_population_config)
                session._adaptive_population_manager = AdaptivePopulationManager(population_config)

            # Pre-compute embeddings for existing ideas if fitness evaluator needs them
            if session.ideas and hasattr(session._fitness_evaluator, 'embeddings_cache'):
                for idea in session.ideas:
                    if idea.content:
                        try:
                            # Generate embedding for idea content
                            embedding = await self.llm_client.embed(idea.content)
                            session._fitness_evaluator.add_embedding(idea.id, embedding)
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for idea {idea.id}: {e}")

            logger.debug(f"Reconstructed components for session {session.id}")

        except Exception as e:
            logger.error(f"Failed to reconstruct session components for {session.id}: {e}")
            raise
