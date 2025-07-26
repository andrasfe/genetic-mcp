"""Session management for genetic MCP server."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from .fitness import FitnessEvaluator
from .genetic_algorithm import GeneticAlgorithm
from .llm_client import MultiModelClient
from .models import (
    EvolutionMode,
    FitnessWeights,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    GeneticParameters,
    Session,
)
from .worker_pool import IdeaGenerator, WorkerPool

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages generation sessions."""

    def __init__(self, llm_client: MultiModelClient,
                 max_sessions_per_client: int = 10,
                 session_timeout_minutes: int = 60):
        self.llm_client = llm_client
        self.max_sessions_per_client = max_sessions_per_client
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions: dict[str, Session] = {}
        self.client_sessions: dict[str, list[str]] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the session manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)

        # Stop all active sessions
        for session in self.sessions.values():
            if hasattr(session, '_worker_pool'):
                await session._worker_pool.stop()

        logger.info("Session manager stopped")

    async def create_session(self, client_id: str, request: GenerationRequest) -> Session:
        """Create a new generation session."""
        # Check client session limit
        client_sessions = self.client_sessions.get(client_id, [])
        if len(client_sessions) >= self.max_sessions_per_client:
            raise ValueError(f"Client {client_id} has reached maximum session limit")

        # Create session
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            client_id=client_id,
            prompt=request.prompt,
            mode=request.mode,
            parameters=request.parameters or GeneticParameters(
                population_size=request.population_size,
                generations=request.generations
            ),
            fitness_weights=request.fitness_weights or FitnessWeights()
        )

        # Initialize session components
        session._worker_pool = WorkerPool(self.llm_client, max_workers=20)
        session._idea_generator = IdeaGenerator(session._worker_pool, self.llm_client)
        session._fitness_evaluator = FitnessEvaluator(session.fitness_weights)
        session._genetic_algorithm = GeneticAlgorithm(session.parameters)

        # Start worker pool
        await session._worker_pool.start()

        # Store session
        self.sessions[session_id] = session
        if client_id not in self.client_sessions:
            self.client_sessions[client_id] = []
        self.client_sessions[client_id].append(session_id)

        logger.info(f"Created session {session_id} for client {client_id}")
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Stop worker pool
        if hasattr(session, '_worker_pool'):
            await session._worker_pool.stop()

        # Remove from client sessions
        if session.client_id in self.client_sessions:
            self.client_sessions[session.client_id].remove(session_id)
            if not self.client_sessions[session.client_id]:
                del self.client_sessions[session.client_id]

        # Remove session
        del self.sessions[session_id]
        logger.info(f"Deleted session {session_id}")

    async def run_generation(self, session: Session, top_k: int = 5) -> GenerationResult:
        """Run the generation process for a session."""
        start_time = datetime.utcnow()

        try:
            # Generate initial population
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

                    # Get selection probabilities
                    probabilities = session._fitness_evaluator.get_selection_probabilities(
                        current_population
                    )

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

                # Get top ideas from final population
                top_ideas = session.get_top_ideas(top_k)

            # Build lineage
            lineage = {}
            for idea in session.ideas:
                if idea.parent_ids:
                    lineage[idea.id] = idea.parent_ids

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

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
            return result

        except Exception as e:
            session.status = "failed"
            logger.error(f"Generation failed for session {session.id}: {e}")
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
