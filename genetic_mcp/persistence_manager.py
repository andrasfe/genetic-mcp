"""Session persistence manager for genetic MCP server."""

import asyncio
import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .logging_config import setup_logging
from .models import FitnessWeights, GeneticParameters, Idea, Session, Worker

logger = setup_logging(component="persistence_manager")


class PersistenceManager:
    """Manages session persistence to SQLite database."""

    def __init__(self, db_path: str = "genetic_mcp_sessions.db"):
        """Initialize persistence manager with database path."""
        self.db_path = db_path
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with self._lock:
            if self._initialized:
                return

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        parameters TEXT NOT NULL,  -- JSON serialized GeneticParameters
                        fitness_weights TEXT NOT NULL,  -- JSON serialized FitnessWeights
                        status TEXT NOT NULL,
                        client_generated BOOLEAN NOT NULL,
                        claude_evaluation_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        claude_evaluation_weight REAL NOT NULL DEFAULT 0.5,
                        adaptive_population_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        adaptive_population_config TEXT NOT NULL DEFAULT '{}',  -- JSON
                        memory_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        parameter_recommendation TEXT NOT NULL DEFAULT '{}',  -- JSON
                        execution_time_seconds REAL NOT NULL DEFAULT 0.0,
                        hybrid_selection_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        selection_strategy TEXT,
                        selection_adaptation_window INTEGER NOT NULL DEFAULT 5,
                        selection_exploration_constant REAL NOT NULL DEFAULT 2.0,
                        selection_min_uses_per_strategy INTEGER NOT NULL DEFAULT 3,
                        selection_performance_history TEXT NOT NULL DEFAULT '{}',  -- JSON
                        advanced_crossover_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        crossover_strategy TEXT,
                        crossover_adaptation_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        crossover_performance_tracking BOOLEAN NOT NULL DEFAULT TRUE,
                        crossover_performance_history TEXT NOT NULL DEFAULT '{}',  -- JSON
                        crossover_config TEXT NOT NULL DEFAULT '{}',  -- JSON
                        intelligent_mutation_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        mutation_strategy TEXT,
                        mutation_adaptation_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        mutation_performance_tracking BOOLEAN NOT NULL DEFAULT TRUE,
                        mutation_performance_history TEXT NOT NULL DEFAULT '{}',  -- JSON
                        mutation_config TEXT NOT NULL DEFAULT '{}',  -- JSON
                        target_embedding BLOB,  -- Compressed pickle of embedding list
                        current_generation INTEGER NOT NULL DEFAULT 0,
                        ideas_per_generation_received TEXT NOT NULL DEFAULT '{}',  -- JSON
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        saved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS ideas (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        generation INTEGER NOT NULL DEFAULT 0,
                        parent_ids TEXT NOT NULL DEFAULT '[]',  -- JSON list
                        scores TEXT NOT NULL DEFAULT '{}',  -- JSON dict
                        fitness REAL NOT NULL DEFAULT 0.0,
                        metadata TEXT NOT NULL DEFAULT '{}',  -- JSON dict
                        created_at TIMESTAMP NOT NULL,
                        claude_evaluation TEXT,  -- JSON dict or NULL
                        claude_score REAL,  -- Claude's score (0-1) or NULL
                        combined_fitness REAL,  -- Combined fitness or NULL
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS workers (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        model TEXT NOT NULL,
                        current_task TEXT,
                        completed_tasks INTEGER NOT NULL DEFAULT 0,
                        failed_tasks INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS session_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        checkpoint_name TEXT NOT NULL,
                        checkpoint_data BLOB NOT NULL,  -- Compressed pickle of additional state
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings_cache (
                        session_id TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB NOT NULL,  -- Compressed pickle of embedding vector
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (session_id, content_hash),
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                """)

                # Create indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON sessions (client_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_saved_at ON sessions (saved_at)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_ideas_session_id ON ideas (session_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_ideas_generation ON ideas (session_id, generation)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_workers_session_id ON workers (session_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON session_checkpoints (session_id, generation)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings_cache (session_id)")

                await db.commit()

            self._initialized = True
            logger.info(f"Persistence manager initialized with database: {self.db_path}")

    async def save_session(self, session: Session, checkpoint_name: str | None = None) -> None:
        """Save a complete session to the database."""
        await self.initialize()

        async with self._lock, aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("BEGIN TRANSACTION")

                    # Save session metadata
                    await self._save_session_metadata(db, session)

                    # Save ideas
                    await self._save_ideas(db, session.id, session.ideas)

                    # Save workers
                    await self._save_workers(db, session.id, session.workers)

                    # Save checkpoint if requested
                    if checkpoint_name:
                        await self._save_checkpoint(db, session, checkpoint_name)

                    # Save embeddings cache if available
                    if hasattr(session, '_fitness_evaluator') and session._fitness_evaluator:
                        await self._save_embeddings_cache(db, session.id, session._fitness_evaluator.embeddings_cache)

                    await db.commit()
                    logger.info(f"Successfully saved session {session.id}" +
                               (f" with checkpoint '{checkpoint_name}'" if checkpoint_name else ""))

                except Exception as e:
                    await db.rollback()
                    logger.error(f"Failed to save session {session.id}: {e}")
                    raise

    async def load_session(self, session_id: str) -> Session | None:
        """Load a complete session from the database."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Load session metadata
            async with db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

            session = await self._deserialize_session_metadata(row)

            # Load ideas
            session.ideas = await self._load_ideas(db, session_id)

            # Load workers
            session.workers = await self._load_workers(db, session_id)

            logger.info(f"Successfully loaded session {session_id} with {len(session.ideas)} ideas and {len(session.workers)} workers")
            return session

    async def list_saved_sessions(self, client_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List saved sessions with basic metadata."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            if client_id:
                query = """
                    SELECT id, client_id, prompt, mode, status, current_generation,
                           execution_time_seconds, created_at, updated_at, saved_at,
                           (SELECT COUNT(*) FROM ideas WHERE session_id = sessions.id) as idea_count
                    FROM sessions
                    WHERE client_id = ?
                    ORDER BY saved_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (client_id, limit, offset)
            else:
                query = """
                    SELECT id, client_id, prompt, mode, status, current_generation,
                           execution_time_seconds, created_at, updated_at, saved_at,
                           (SELECT COUNT(*) FROM ideas WHERE session_id = sessions.id) as idea_count
                    FROM sessions
                    ORDER BY saved_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (limit, offset)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            sessions = []
            for row in rows:
                sessions.append({
                    "id": row[0],
                    "client_id": row[1],
                    "prompt": row[2][:200] + "..." if len(row[2]) > 200 else row[2],  # Truncate prompt
                    "mode": row[3],
                    "status": row[4],
                    "current_generation": row[5],
                    "execution_time_seconds": row[6],
                    "created_at": row[7],
                    "updated_at": row[8],
                    "saved_at": row[9],
                    "idea_count": row[10]
                })

            return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data."""
        await self.initialize()

        async with self._lock, aiosqlite.connect(self.db_path) as db:
                # Check if session exists
                async with db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)) as cursor:
                    if not await cursor.fetchone():
                        return False

                # Delete session (CASCADE will handle related tables)
                await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                await db.commit()

                logger.info(f"Successfully deleted session {session_id}")
                return True

    async def save_checkpoint(self, session: Session, checkpoint_name: str, additional_data: dict[str, Any] | None = None) -> None:
        """Save a checkpoint for a session."""
        await self.initialize()

        checkpoint_data = {
            "generation": session.current_generation,
            "status": session.status,
            "timestamp": datetime.utcnow().isoformat(),
            "additional_data": additional_data or {}
        }

        # Add fitness evaluator state if available
        if hasattr(session, '_fitness_evaluator') and session._fitness_evaluator:
            checkpoint_data["embeddings_cache_size"] = len(session._fitness_evaluator.embeddings_cache)

        # Add genetic algorithm state if available
        if hasattr(session, '_genetic_algorithm') and session._genetic_algorithm:
            checkpoint_data["genetic_algorithm_state"] = {
                "current_generation": session.current_generation,
                "parameters": session.parameters.model_dump()
            }

        compressed_data = gzip.compress(pickle.dumps(checkpoint_data))

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO session_checkpoints (session_id, generation, checkpoint_name, checkpoint_data)
                VALUES (?, ?, ?, ?)
            """, (session.id, session.current_generation, checkpoint_name, compressed_data))
            await db.commit()

        logger.info(f"Saved checkpoint '{checkpoint_name}' for session {session.id} at generation {session.current_generation}")

    async def load_checkpoint(self, session_id: str, checkpoint_name: str) -> dict[str, Any] | None:
        """Load a specific checkpoint for a session."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT checkpoint_data FROM session_checkpoints
                WHERE session_id = ? AND checkpoint_name = ?
                ORDER BY created_at DESC LIMIT 1
            """, (session_id, checkpoint_name)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

            checkpoint_data = pickle.loads(gzip.decompress(row[0]))
            return checkpoint_data

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a session."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db, db.execute("""
                SELECT checkpoint_name, generation, created_at
                FROM session_checkpoints
                WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "checkpoint_name": row[0],
                "generation": row[1],
                "created_at": row[2]
            }
            for row in rows
        ]

    async def _save_session_metadata(self, db: aiosqlite.Connection, session: Session) -> None:
        """Save session metadata to the database."""
        # Serialize target_embedding if present
        target_embedding_blob = None
        if session.target_embedding:
            target_embedding_blob = gzip.compress(pickle.dumps(session.target_embedding))

        await db.execute("""
            INSERT OR REPLACE INTO sessions (
                id, client_id, prompt, mode, parameters, fitness_weights, status,
                client_generated, claude_evaluation_enabled, claude_evaluation_weight,
                adaptive_population_enabled, adaptive_population_config, memory_enabled,
                parameter_recommendation, execution_time_seconds, hybrid_selection_enabled,
                selection_strategy, selection_adaptation_window, selection_exploration_constant,
                selection_min_uses_per_strategy, selection_performance_history,
                advanced_crossover_enabled, crossover_strategy, crossover_adaptation_enabled,
                crossover_performance_tracking, crossover_performance_history, crossover_config,
                intelligent_mutation_enabled, mutation_strategy, mutation_adaptation_enabled,
                mutation_performance_tracking, mutation_performance_history, mutation_config,
                target_embedding, current_generation, ideas_per_generation_received,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id, session.client_id, session.prompt, session.mode.value,
            json.dumps(session.parameters.model_dump()),
            json.dumps(session.fitness_weights.model_dump()),
            session.status, session.client_generated, session.claude_evaluation_enabled,
            session.claude_evaluation_weight, session.adaptive_population_enabled,
            json.dumps(session.adaptive_population_config), session.memory_enabled,
            json.dumps(session.parameter_recommendation), session.execution_time_seconds,
            session.hybrid_selection_enabled, session.selection_strategy,
            session.selection_adaptation_window, session.selection_exploration_constant,
            session.selection_min_uses_per_strategy,
            json.dumps(session.selection_performance_history),
            session.advanced_crossover_enabled, session.crossover_strategy,
            session.crossover_adaptation_enabled, session.crossover_performance_tracking,
            json.dumps(session.crossover_performance_history),
            json.dumps(session.crossover_config), session.intelligent_mutation_enabled,
            session.mutation_strategy, session.mutation_adaptation_enabled,
            session.mutation_performance_tracking,
            json.dumps(session.mutation_performance_history),
            json.dumps(session.mutation_config), target_embedding_blob,
            session.current_generation, json.dumps(session.ideas_per_generation_received),
            session.created_at.isoformat(), session.updated_at.isoformat()
        ))

    async def _save_ideas(self, db: aiosqlite.Connection, session_id: str, ideas: list[Idea]) -> None:
        """Save ideas to the database."""
        # Clear existing ideas for this session
        await db.execute("DELETE FROM ideas WHERE session_id = ?", (session_id,))

        # Insert all ideas
        for idea in ideas:
            await db.execute("""
                INSERT INTO ideas (
                    id, session_id, content, generation, parent_ids, scores, fitness,
                    metadata, created_at, claude_evaluation, claude_score, combined_fitness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idea.id, session_id, idea.content, idea.generation,
                json.dumps(idea.parent_ids), json.dumps(idea.scores),
                idea.fitness, json.dumps(idea.metadata), idea.created_at.isoformat(),
                json.dumps(idea.claude_evaluation) if idea.claude_evaluation else None,
                idea.claude_score, idea.combined_fitness
            ))

    async def _save_workers(self, db: aiosqlite.Connection, session_id: str, workers: list[Worker]) -> None:
        """Save workers to the database."""
        # Clear existing workers for this session
        await db.execute("DELETE FROM workers WHERE session_id = ?", (session_id,))

        # Insert all workers
        for worker in workers:
            await db.execute("""
                INSERT INTO workers (
                    id, session_id, status, model, current_task, completed_tasks,
                    failed_tasks, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                worker.id, session_id, worker.status.value, worker.model,
                worker.current_task, worker.completed_tasks, worker.failed_tasks,
                worker.created_at.isoformat()
            ))

    async def _save_checkpoint(self, db: aiosqlite.Connection, session: Session, checkpoint_name: str) -> None:
        """Save a checkpoint to the database."""
        checkpoint_data = {
            "session_id": session.id,
            "generation": session.current_generation,
            "status": session.status,
            "timestamp": datetime.utcnow().isoformat()
        }

        compressed_data = gzip.compress(pickle.dumps(checkpoint_data))

        await db.execute("""
            INSERT INTO session_checkpoints (session_id, generation, checkpoint_name, checkpoint_data)
            VALUES (?, ?, ?, ?)
        """, (session.id, session.current_generation, checkpoint_name, compressed_data))

    async def _save_embeddings_cache(self, db: aiosqlite.Connection, session_id: str, embeddings_cache: dict[str, Any]) -> None:
        """Save embeddings cache to the database."""
        # Clear existing embeddings for this session
        await db.execute("DELETE FROM embeddings_cache WHERE session_id = ?", (session_id,))

        # Insert embeddings
        for content_hash, embedding in embeddings_cache.items():
            compressed_embedding = gzip.compress(pickle.dumps(embedding))
            await db.execute("""
                INSERT INTO embeddings_cache (session_id, content_hash, embedding)
                VALUES (?, ?, ?)
            """, (session_id, content_hash, compressed_embedding))

    async def _load_ideas(self, db: aiosqlite.Connection, session_id: str) -> list[Idea]:
        """Load ideas from the database."""
        async with db.execute("""
            SELECT id, content, generation, parent_ids, scores, fitness, metadata,
                   created_at, claude_evaluation, claude_score, combined_fitness
            FROM ideas WHERE session_id = ? ORDER BY generation, created_at
        """, (session_id,)) as cursor:
            rows = await cursor.fetchall()

        ideas = []
        for row in rows:
            idea = Idea(
                id=row[0],
                content=row[1],
                generation=row[2],
                parent_ids=json.loads(row[3]),
                scores=json.loads(row[4]),
                fitness=row[5],
                metadata=json.loads(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                claude_evaluation=json.loads(row[8]) if row[8] else None,
                claude_score=row[9],
                combined_fitness=row[10]
            )
            ideas.append(idea)

        return ideas

    async def _load_workers(self, db: aiosqlite.Connection, session_id: str) -> list[Worker]:
        """Load workers from the database."""
        from .models import WorkerStatus

        async with db.execute("""
            SELECT id, status, model, current_task, completed_tasks, failed_tasks, created_at
            FROM workers WHERE session_id = ?
        """, (session_id,)) as cursor:
            rows = await cursor.fetchall()

        workers = []
        for row in rows:
            worker = Worker(
                id=row[0],
                status=WorkerStatus(row[1]),
                model=row[2],
                current_task=row[3],
                completed_tasks=row[4],
                failed_tasks=row[5],
                created_at=datetime.fromisoformat(row[6])
            )
            workers.append(worker)

        return workers

    async def _deserialize_session_metadata(self, row) -> Session:
        """Deserialize session metadata from database row."""
        from .models import EvolutionMode

        # Handle target_embedding
        target_embedding = None
        if row[33]:  # target_embedding blob
            target_embedding = pickle.loads(gzip.decompress(row[33]))

        session = Session(
            id=row[0],
            client_id=row[1],
            prompt=row[2],
            mode=EvolutionMode(row[3]),
            parameters=GeneticParameters(**json.loads(row[4])),
            fitness_weights=FitnessWeights(**json.loads(row[5])),
            status=row[6],
            client_generated=bool(row[7]),
            claude_evaluation_enabled=bool(row[8]),
            claude_evaluation_weight=row[9],
            adaptive_population_enabled=bool(row[10]),
            adaptive_population_config=json.loads(row[11]),
            memory_enabled=bool(row[12]),
            parameter_recommendation=json.loads(row[13]),
            execution_time_seconds=row[14],
            hybrid_selection_enabled=bool(row[15]),
            selection_strategy=row[16],
            selection_adaptation_window=row[17],
            selection_exploration_constant=row[18],
            selection_min_uses_per_strategy=row[19],
            selection_performance_history=json.loads(row[20]),
            advanced_crossover_enabled=bool(row[21]),
            crossover_strategy=row[22],
            crossover_adaptation_enabled=bool(row[23]),
            crossover_performance_tracking=bool(row[24]),
            crossover_performance_history=json.loads(row[25]),
            crossover_config=json.loads(row[26]),
            intelligent_mutation_enabled=bool(row[27]),
            mutation_strategy=row[28],
            mutation_adaptation_enabled=bool(row[29]),
            mutation_performance_tracking=bool(row[30]),
            mutation_performance_history=json.loads(row[31]),
            mutation_config=json.loads(row[32]),
            target_embedding=target_embedding,
            current_generation=row[34],
            ideas_per_generation_received=json.loads(row[35]),
            created_at=datetime.fromisoformat(row[36]),
            updated_at=datetime.fromisoformat(row[37]),
        )

        return session

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Delete old sessions (CASCADE will handle related tables)
            async with db.execute(f"""
                DELETE FROM sessions
                WHERE datetime(saved_at) < datetime('now', '-{days_old} days')
            """) as cursor:
                rows_affected = cursor.rowcount

            await db.commit()

            if rows_affected > 0:
                logger.info(f"Cleaned up {rows_affected} sessions older than {days_old} days")

            return rows_affected

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            stats = {}

            # Count tables
            for table in ["sessions", "ideas", "workers", "session_checkpoints", "embeddings_cache"]:
                async with db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                    count = await cursor.fetchone()
                    stats[f"{table}_count"] = count[0]

            # Get database size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats["database_size_bytes"] = db_path.stat().st_size

            # Get oldest and newest sessions
            async with db.execute("SELECT MIN(created_at), MAX(created_at) FROM sessions") as cursor:
                row = await cursor.fetchone()
                if row[0]:
                    stats["oldest_session"] = row[0]
                    stats["newest_session"] = row[1]

            return stats
