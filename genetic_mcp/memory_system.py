"""Memory & Learning System for genetic-mcp.

This module implements a persistent learning system that analyzes successful evolution patterns
and provides parameter recommendations for future sessions based on historical performance.
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .embedding_providers import get_embedding_provider
from .models import FitnessWeights, GeneticParameters, Session

logger = logging.getLogger(__name__)


class PromptCategory(BaseModel):
    """Categorized prompt information."""
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)


class EvolutionPattern(BaseModel):
    """Represents a successful evolution pattern."""
    id: str
    session_id: str
    prompt: str
    prompt_category: str
    parameters: GeneticParameters
    fitness_weights: FitnessWeights
    success_metrics: dict[str, float]
    final_fitness_scores: list[float]
    convergence_generation: int
    diversity_maintained: float
    execution_time_seconds: float
    ideas_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OperationEffectiveness(BaseModel):
    """Tracks effectiveness of specific genetic operations."""
    operation_type: str  # 'crossover', 'mutation', 'selection'
    context: str  # Prompt category or specific context
    effectiveness_score: float = Field(ge=0.0, le=1.0)
    usage_count: int = 0
    success_rate: float = 0.0
    avg_fitness_improvement: float = 0.0


class ParameterRecommendation(BaseModel):
    """Parameter recommendation based on historical data."""
    parameters: GeneticParameters
    fitness_weights: FitnessWeights
    confidence: float = Field(ge=0.0, le=1.0)
    similar_sessions_count: int
    expected_performance: dict[str, float]
    reasoning: str


class PromptCategorizer:
    """Categorizes prompts into different types for better parameter matching."""

    CATEGORIES = {
        "code_generation": [
            "code", "programming", "function", "algorithm", "implementation",
            "debug", "refactor", "software", "development", "script"
        ],
        "creative_writing": [
            "story", "creative", "writing", "narrative", "character", "plot",
            "fiction", "poem", "dialogue", "scene"
        ],
        "business_ideas": [
            "business", "startup", "product", "service", "market", "customer",
            "revenue", "strategy", "opportunity", "venture"
        ],
        "problem_solving": [
            "problem", "solution", "solve", "challenge", "issue", "approach",
            "method", "strategy", "technique", "fix"
        ],
        "research_analysis": [
            "research", "analysis", "study", "investigate", "examine", "explore",
            "findings", "data", "hypothesis", "conclusion"
        ],
        "design_concepts": [
            "design", "concept", "interface", "user", "experience", "visual",
            "layout", "aesthetic", "prototype", "mockup"
        ]
    }

    def categorize_prompt(self, prompt: str) -> PromptCategory:
        """Categorize a prompt based on keywords."""
        prompt_lower = prompt.lower()
        category_scores: dict[str, dict[str, object]] = {}

        for category, keywords in self.CATEGORIES.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                category_scores[category] = {
                    'score': score,
                    'keywords': matched_keywords
                }

        if not category_scores:
            return PromptCategory(
                category="general",
                confidence=0.5,
                keywords=[]
            )

        best_category = max(category_scores.keys(), key=lambda k: float(category_scores[k]['score']))
        max_score = float(category_scores[best_category]['score'])
        confidence = min(max_score / 3.0, 1.0)  # Normalize confidence

        return PromptCategory(
            category=best_category,
            confidence=confidence,
            keywords=list(category_scores[best_category]['keywords'])
        )


class MemoryDatabase:
    """SQLite database for storing evolution patterns and learning data."""

    def __init__(self, db_path: str = "genetic_mcp_memory.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS evolution_patterns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                prompt_category TEXT NOT NULL,
                prompt_embedding BLOB,
                parameters_json TEXT NOT NULL,
                fitness_weights_json TEXT NOT NULL,
                success_metrics_json TEXT NOT NULL,
                final_fitness_scores_json TEXT NOT NULL,
                convergence_generation INTEGER NOT NULL,
                diversity_maintained REAL NOT NULL,
                execution_time_seconds REAL NOT NULL,
                ideas_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS operation_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                context TEXT NOT NULL,
                effectiveness_score REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_fitness_improvement REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prompt_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_evolution_patterns_category ON evolution_patterns(prompt_category);
            CREATE INDEX IF NOT EXISTS idx_evolution_patterns_created_at ON evolution_patterns(created_at);
            CREATE INDEX IF NOT EXISTS idx_operation_effectiveness_context ON operation_effectiveness(context);
            CREATE INDEX IF NOT EXISTS idx_prompt_embeddings_category ON prompt_embeddings(category);
            """)

    def store_evolution_pattern(self, pattern: EvolutionPattern, embedding: list[float]) -> None:
        """Store a successful evolution pattern."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evolution_patterns
                (id, session_id, prompt, prompt_category, prompt_embedding,
                 parameters_json, fitness_weights_json, success_metrics_json,
                 final_fitness_scores_json, convergence_generation, diversity_maintained,
                 execution_time_seconds, ideas_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id,
                pattern.session_id,
                pattern.prompt,
                pattern.prompt_category,
                self._encode_embedding(embedding),
                json.dumps(pattern.parameters.model_dump()),
                json.dumps(pattern.fitness_weights.model_dump()),
                json.dumps(pattern.success_metrics),
                json.dumps(pattern.final_fitness_scores),
                pattern.convergence_generation,
                pattern.diversity_maintained,
                pattern.execution_time_seconds,
                pattern.ideas_count,
                pattern.created_at.isoformat()
            ))

    def get_similar_patterns(self, target_embedding: list[float],
                           category: str, limit: int = 10) -> list[tuple[EvolutionPattern, float]]:
        """Get evolution patterns similar to the target embedding."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM evolution_patterns
                WHERE prompt_category = ?
                ORDER BY created_at DESC
                LIMIT 50
            """, (category,))

            patterns = []
            target_embedding_np = np.array(target_embedding)

            for row in cursor.fetchall():
                stored_embedding = self._decode_embedding(row[4])  # prompt_embedding
                if stored_embedding is not None:
                    similarity = self._cosine_similarity(target_embedding_np, stored_embedding)

                    pattern = EvolutionPattern(
                        id=row[0],
                        session_id=row[1],
                        prompt=row[2],
                        prompt_category=row[3],
                        parameters=GeneticParameters(**json.loads(row[5])),
                        fitness_weights=FitnessWeights(**json.loads(row[6])),
                        success_metrics=json.loads(row[7]),
                        final_fitness_scores=json.loads(row[8]),
                        convergence_generation=row[9],
                        diversity_maintained=row[10],
                        execution_time_seconds=row[11],
                        ideas_count=row[12],
                        created_at=datetime.fromisoformat(row[13])
                    )
                    patterns.append((pattern, similarity))

            # Sort by similarity and return top results
            patterns.sort(key=lambda x: x[1], reverse=True)
            return patterns[:limit]

    def store_operation_effectiveness(self, operation: OperationEffectiveness) -> None:
        """Store or update operation effectiveness data."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if record exists
            cursor = conn.execute("""
                SELECT id, usage_count FROM operation_effectiveness
                WHERE operation_type = ? AND context = ?
            """, (operation.operation_type, operation.context))

            existing = cursor.fetchone()

            if existing:
                # Update existing record
                new_usage_count = existing[1] + operation.usage_count
                conn.execute("""
                    UPDATE operation_effectiveness
                    SET effectiveness_score = ?, usage_count = ?,
                        success_rate = ?, avg_fitness_improvement = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    operation.effectiveness_score,
                    new_usage_count,
                    operation.success_rate,
                    operation.avg_fitness_improvement,
                    existing[0]
                ))
            else:
                # Insert new record
                conn.execute("""
                    INSERT INTO operation_effectiveness
                    (operation_type, context, effectiveness_score, usage_count,
                     success_rate, avg_fitness_improvement)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    operation.operation_type,
                    operation.context,
                    operation.effectiveness_score,
                    operation.usage_count,
                    operation.success_rate,
                    operation.avg_fitness_improvement
                ))

    def get_operation_effectiveness(self, operation_type: str, context: str) -> OperationEffectiveness | None:
        """Get operation effectiveness data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT operation_type, context, effectiveness_score, usage_count,
                       success_rate, avg_fitness_improvement
                FROM operation_effectiveness
                WHERE operation_type = ? AND context = ?
            """, (operation_type, context))

            row = cursor.fetchone()
            if row:
                return OperationEffectiveness(
                    operation_type=row[0],
                    context=row[1],
                    effectiveness_score=row[2],
                    usage_count=row[3],
                    success_rate=row[4],
                    avg_fitness_improvement=row[5]
                )
        return None

    def get_category_statistics(self, category: str, days: int = 30) -> dict[str, Any]:
        """Get statistics for a specific category."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as session_count,
                       AVG(diversity_maintained) as avg_diversity,
                       AVG(convergence_generation) as avg_convergence,
                       AVG(execution_time_seconds) as avg_execution_time,
                       AVG(ideas_count) as avg_ideas_count
                FROM evolution_patterns
                WHERE prompt_category = ? AND created_at > ?
            """, (category, cutoff_date.isoformat()))

            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "session_count": row[0],
                    "avg_diversity": row[1] or 0.0,
                    "avg_convergence": row[2] or 0.0,
                    "avg_execution_time": row[3] or 0.0,
                    "avg_ideas_count": row[4] or 0.0
                }

        return {
            "session_count": 0,
            "avg_diversity": 0.0,
            "avg_convergence": 0.0,
            "avg_execution_time": 0.0,
            "avg_ideas_count": 0.0
        }

    def _encode_embedding(self, embedding: list[float]) -> bytes:
        """Encode embedding as bytes for storage."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _decode_embedding(self, data: bytes) -> np.ndarray | None:
        """Decode embedding from bytes."""
        try:
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to decode embedding: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0


class ParameterOptimizer:
    """Optimizes parameters based on historical performance data."""

    def __init__(self, memory_db: MemoryDatabase):
        self.memory_db = memory_db

    def recommend_parameters(self, prompt: str, category: PromptCategory,
                           target_embedding: list[float]) -> ParameterRecommendation:
        """Generate parameter recommendations based on similar past sessions."""
        # Get similar patterns
        similar_patterns = self.memory_db.get_similar_patterns(
            target_embedding, category.category, limit=20
        )

        if not similar_patterns:
            return self._default_recommendation(category)

        # Filter by similarity threshold
        high_similarity_patterns = [
            (pattern, similarity) for pattern, similarity in similar_patterns
            if similarity > 0.7  # Only consider high-similarity patterns
        ]

        if not high_similarity_patterns:
            return self._fallback_recommendation(similar_patterns, category)

        # Analyze successful patterns
        return self._analyze_successful_patterns(high_similarity_patterns, category)

    def _analyze_successful_patterns(self, patterns: list[tuple[EvolutionPattern, float]],
                                   category: PromptCategory) -> ParameterRecommendation:
        """Analyze patterns to generate optimal parameters."""
        # Extract patterns and weights
        pattern_list = [p[0] for p in patterns]
        similarities = [p[1] for p in patterns]

        # Weight by similarity and success metrics
        weights = []
        for pattern, similarity in patterns:
            success_score = self._calculate_success_score(pattern)
            weight = similarity * success_score
            weights.append(weight)

        weights_array = np.array(weights)
        weights = weights_array / weights_array.sum()  # Normalize

        # Calculate weighted averages
        population_sizes = [p.parameters.population_size for p in pattern_list]
        generations = [p.parameters.generations for p in pattern_list]
        mutation_rates = [p.parameters.mutation_rate for p in pattern_list]
        crossover_rates = [p.parameters.crossover_rate for p in pattern_list]
        elitism_counts = [p.parameters.elitism_count for p in pattern_list]

        # Fitness weights
        relevance_weights = [p.fitness_weights.relevance for p in pattern_list]
        novelty_weights = [p.fitness_weights.novelty for p in pattern_list]
        feasibility_weights = [p.fitness_weights.feasibility for p in pattern_list]

        # Calculate weighted recommendations
        recommended_params = GeneticParameters(
            population_size=max(2, int(np.average(population_sizes, weights=weights))),
            generations=max(1, int(np.average(generations, weights=weights))),
            mutation_rate=max(0.0, min(1.0, float(np.average(mutation_rates, weights=weights)))),
            crossover_rate=max(0.0, min(1.0, float(np.average(crossover_rates, weights=weights)))),
            elitism_count=max(0, int(np.average(elitism_counts, weights=weights)))
        )

        # Normalize fitness weights to sum to 1
        avg_relevance = float(np.average(relevance_weights, weights=weights))
        avg_novelty = float(np.average(novelty_weights, weights=weights))
        avg_feasibility = float(np.average(feasibility_weights, weights=weights))

        total_weight = avg_relevance + avg_novelty + avg_feasibility
        if total_weight > 0:
            recommended_fitness = FitnessWeights(
                relevance=avg_relevance / total_weight,
                novelty=avg_novelty / total_weight,
                feasibility=avg_feasibility / total_weight
            )
        else:
            recommended_fitness = FitnessWeights()

        # Calculate confidence
        confidence = min(0.95, category.confidence * len(patterns) / 10.0)

        # Expected performance
        expected_performance = {
            "avg_fitness": float(np.average([max(p.final_fitness_scores) for p in pattern_list], weights=weights)),
            "convergence_generation": float(np.average([p.convergence_generation for p in pattern_list], weights=weights)),
            "diversity_maintained": float(np.average([p.diversity_maintained for p in pattern_list], weights=weights)),
            "execution_time": float(np.average([p.execution_time_seconds for p in pattern_list], weights=weights))
        }

        # Generate reasoning
        reasoning = (
            f"Based on {len(patterns)} similar {category.category} sessions with "
            f"average similarity of {np.mean(similarities):.2f}. "
            f"Expected to achieve {expected_performance['avg_fitness']:.2f} fitness "
            f"with convergence around generation {expected_performance['convergence_generation']:.0f}."
        )

        return ParameterRecommendation(
            parameters=recommended_params,
            fitness_weights=recommended_fitness,
            confidence=confidence,
            similar_sessions_count=len(patterns),
            expected_performance=expected_performance,
            reasoning=reasoning
        )

    def _calculate_success_score(self, pattern: EvolutionPattern) -> float:
        """Calculate a success score for a pattern."""
        # Factors to consider:
        # 1. Final fitness scores (higher is better)
        # 2. Convergence speed (faster is better, up to a point)
        # 3. Diversity maintained (higher is better)
        # 4. Ideas generated (more diverse exploration)

        max_fitness = max(pattern.final_fitness_scores) if pattern.final_fitness_scores else 0.0
        avg_fitness = sum(pattern.final_fitness_scores) / len(pattern.final_fitness_scores) if pattern.final_fitness_scores else 0.0

        # Normalize convergence (prefer 2-8 generations, penalize too fast/slow)
        convergence_score = 1.0
        if pattern.convergence_generation < 2:
            convergence_score = 0.7  # Too fast, might have missed better solutions
        elif pattern.convergence_generation > 10:
            convergence_score = 0.8  # Too slow

        # Combine factors
        success_score = (
            max_fitness * 0.4 +
            avg_fitness * 0.3 +
            pattern.diversity_maintained * 0.2 +
            convergence_score * 0.1
        )

        return min(1.0, max(0.0, success_score))

    def _default_recommendation(self, category: PromptCategory) -> ParameterRecommendation:
        """Provide default parameters when no historical data is available."""
        # Category-specific defaults
        category_defaults = {
            "code_generation": {
                "population_size": 15,
                "generations": 6,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "fitness_weights": FitnessWeights(relevance=0.5, novelty=0.2, feasibility=0.3)
            },
            "creative_writing": {
                "population_size": 20,
                "generations": 8,
                "mutation_rate": 0.2,
                "crossover_rate": 0.7,
                "fitness_weights": FitnessWeights(relevance=0.3, novelty=0.5, feasibility=0.2)
            },
            "business_ideas": {
                "population_size": 12,
                "generations": 5,
                "mutation_rate": 0.1,
                "crossover_rate": 0.75,
                "fitness_weights": FitnessWeights(relevance=0.4, novelty=0.3, feasibility=0.3)
            }
        }

        defaults = category_defaults.get(category.category, {
            "population_size": 10,
            "generations": 5,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "fitness_weights": FitnessWeights()
        })

        return ParameterRecommendation(
            parameters=GeneticParameters(
                population_size=int(defaults["population_size"]),
                generations=int(defaults["generations"]),
                mutation_rate=float(defaults["mutation_rate"]),
                crossover_rate=float(defaults["crossover_rate"]),
                elitism_count=max(1, int(defaults["population_size"]) // 8)
            ),
            fitness_weights=FitnessWeights() if not isinstance(defaults["fitness_weights"], FitnessWeights) else defaults["fitness_weights"],
            confidence=0.6,
            similar_sessions_count=0,
            expected_performance={
                "avg_fitness": 0.7,
                "convergence_generation": 3,
                "diversity_maintained": 0.6,
                "execution_time": 60.0
            },
            reasoning=f"Default parameters for {category.category} category (no historical data available)."
        )

    def _fallback_recommendation(self, patterns: list[tuple[EvolutionPattern, float]],
                               category: PromptCategory) -> ParameterRecommendation:
        """Fallback when similarity is low but some patterns exist."""
        # Use patterns but with lower confidence
        result = self._analyze_successful_patterns(patterns, category)
        result.confidence *= 0.7  # Reduce confidence
        result.reasoning += " (Low similarity fallback)"
        return result


class MemorySystem:
    """Main memory and learning system for genetic-mcp."""

    def __init__(self, db_path: str = "genetic_mcp_memory.db",
                 enable_learning: bool = True):
        self.enable_learning = enable_learning
        self.memory_db = MemoryDatabase(db_path) if enable_learning else None
        self.categorizer = PromptCategorizer()
        self.optimizer = ParameterOptimizer(self.memory_db) if enable_learning and self.memory_db else None
        self._embedding_cache: dict[str, list[float]] = {}

        logger.info(f"Memory system initialized (learning={'enabled' if enable_learning else 'disabled'})")

    async def get_parameter_recommendation(self, prompt: str) -> ParameterRecommendation | None:
        """Get parameter recommendations for a new session."""
        if not self.enable_learning or not self.optimizer:
            return None

        try:
            # Categorize the prompt
            category = self.categorizer.categorize_prompt(prompt)

            # Get or generate embedding
            embedding = await self._get_embedding(prompt)

            # Get recommendation
            recommendation = self.optimizer.recommend_parameters(prompt, category, embedding)

            logger.info(f"Generated parameter recommendation for '{category.category}' category "
                       f"with confidence {recommendation.confidence:.2f}")

            return recommendation

        except Exception as e:
            logger.error(f"Failed to generate parameter recommendation: {e}")
            return None

    async def store_session_results(self, session: Session) -> bool:
        """Store successful session results for future learning."""
        if not self.enable_learning or not self.memory_db:
            return False

        try:
            # Only store completed sessions with reasonable performance
            if session.status != "completed" or not session.ideas:
                return False

            # Calculate success metrics
            top_ideas = session.get_top_ideas(5)
            if not top_ideas:
                return False

            final_fitness_scores = [idea.fitness for idea in top_ideas]
            max_fitness = max(final_fitness_scores)

            # Only store if performance is above threshold
            if max_fitness < 0.5:
                logger.debug(f"Session {session.id} fitness too low ({max_fitness:.2f}), not storing")
                return False

            # Categorize prompt and get embedding
            category = self.categorizer.categorize_prompt(session.prompt)
            embedding = await self._get_embedding(session.prompt)

            # Calculate diversity
            diversity_maintained = self._calculate_diversity(session.ideas)

            # Find convergence generation (when fitness stopped improving significantly)
            convergence_generation = self._find_convergence_generation(session.ideas)

            # Create evolution pattern
            pattern = EvolutionPattern(
                id=f"pattern_{session.id}",
                session_id=session.id,
                prompt=session.prompt,
                prompt_category=category.category,
                parameters=session.parameters,
                fitness_weights=session.fitness_weights,
                success_metrics={
                    "max_fitness": max_fitness,
                    "avg_top_fitness": sum(final_fitness_scores) / len(final_fitness_scores),
                    "improvement_rate": self._calculate_improvement_rate(session.ideas),
                    "category_confidence": category.confidence
                },
                final_fitness_scores=final_fitness_scores,
                convergence_generation=convergence_generation,
                diversity_maintained=diversity_maintained,
                execution_time_seconds=getattr(session, 'execution_time_seconds', 0.0),
                ideas_count=len(session.ideas)
            )

            # Store pattern
            self.memory_db.store_evolution_pattern(pattern, embedding)

            # Store operation effectiveness
            await self._store_operation_effectiveness(session, category.category)

            logger.info(f"Stored evolution pattern for session {session.id} "
                       f"(category: {category.category}, fitness: {max_fitness:.2f})")

            return True

        except Exception as e:
            logger.error(f"Failed to store session results: {e}")
            return False

    async def get_category_insights(self, category: str, days: int = 30) -> dict[str, Any]:
        """Get insights and statistics for a specific category."""
        if not self.enable_learning or not self.memory_db:
            return {}

        try:
            stats = self.memory_db.get_category_statistics(category, days)
            return {
                "category": category,
                "period_days": days,
                "statistics": stats,
                "insights": self._generate_insights(stats, category)
            }
        except Exception as e:
            logger.error(f"Failed to get category insights: {e}")
            return {}

    def get_memory_status(self) -> dict[str, Any]:
        """Get current memory system status."""
        status = {
            "enabled": self.enable_learning,
            "database_path": getattr(self.memory_db, 'db_path', None) if self.memory_db else None,
            "cache_size": len(self._embedding_cache),
            "categorizer_categories": list(self.categorizer.CATEGORIES.keys())
        }

        if self.memory_db:
            try:
                with sqlite3.connect(self.memory_db.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM evolution_patterns")
                    status["stored_patterns"] = cursor.fetchone()[0]

                    cursor = conn.execute("SELECT COUNT(*) FROM operation_effectiveness")
                    status["operation_records"] = cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Failed to get memory status: {e}")
                status["error"] = str(e)

        return status

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, using cache when possible."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            embedding_provider = get_embedding_provider()
            embedding = await embedding_provider.embed(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return dummy embedding as fallback
            return [0.0] * 384

    def _calculate_diversity(self, ideas: list) -> float:
        """Calculate diversity score for a set of ideas."""
        if len(ideas) < 2:
            return 0.0

        # Simple diversity calculation based on content length variation
        # In a real implementation, this would use embeddings and clustering
        contents = [idea.content for idea in ideas]
        lengths = [len(content) for content in contents]

        if not lengths:
            return 0.0

        # Use coefficient of variation as a simple diversity measure
        mean_length = sum(lengths) / len(lengths)
        if mean_length == 0:
            return 0.0

        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        std_dev = variance ** 0.5
        diversity = std_dev / mean_length

        return float(min(1.0, diversity))

    def _find_convergence_generation(self, ideas: list) -> int:
        """Find the generation where fitness converged."""
        if not ideas:
            return 0

        # Group ideas by generation
        generation_fitness: dict[int, list[float]] = {}
        for idea in ideas:
            gen = getattr(idea, 'generation', 0)
            if gen not in generation_fitness:
                generation_fitness[gen] = []
            generation_fitness[gen].append(getattr(idea, 'fitness', 0.0))

        if not generation_fitness:
            return 0

        # Find when max fitness stopped improving significantly
        generations = sorted(generation_fitness.keys())
        max_fitness_per_gen = [max(generation_fitness[gen]) for gen in generations]

        convergence_gen = generations[-1]  # Default to last generation
        improvement_threshold = 0.01

        for i in range(1, len(max_fitness_per_gen)):
            improvement = max_fitness_per_gen[i] - max_fitness_per_gen[i-1]
            if improvement < improvement_threshold:
                convergence_gen = generations[i]
                break

        return int(convergence_gen)

    def _calculate_improvement_rate(self, ideas: list) -> float:
        """Calculate the rate of improvement across generations."""
        if not ideas:
            return 0.0

        # Group by generation and calculate improvement
        generation_fitness: dict[int, list[float]] = {}
        for idea in ideas:
            gen = getattr(idea, 'generation', 0)
            if gen not in generation_fitness:
                generation_fitness[gen] = []
            generation_fitness[gen].append(getattr(idea, 'fitness', 0.0))

        generations = sorted(generation_fitness.keys())
        if len(generations) < 2:
            return 0.0

        first_gen_max = max(generation_fitness[generations[0]])
        last_gen_max = max(generation_fitness[generations[-1]])

        improvement = last_gen_max - first_gen_max
        return float(max(0.0, improvement))

    async def _store_operation_effectiveness(self, session: Session, category: str) -> None:
        """Store effectiveness data for genetic operations."""
        if not self.memory_db:
            return

        # Calculate operation effectiveness based on session results
        ideas = session.ideas
        if not ideas:
            return

        # Analyze crossover effectiveness
        crossover_ideas = [i for i in ideas if len(getattr(i, 'parent_ids', [])) >= 2]
        if crossover_ideas:
            avg_crossover_fitness = sum(getattr(i, 'fitness', 0.0) for i in crossover_ideas) / len(crossover_ideas)
            crossover_op = OperationEffectiveness(
                operation_type="crossover",
                context=category,
                effectiveness_score=min(1.0, avg_crossover_fitness),
                usage_count=len(crossover_ideas),
                success_rate=len([i for i in crossover_ideas if getattr(i, 'fitness', 0.0) > 0.5]) / len(crossover_ideas),
                avg_fitness_improvement=avg_crossover_fitness
            )
            self.memory_db.store_operation_effectiveness(crossover_op)

        # Analyze mutation effectiveness
        mutation_ideas = [i for i in ideas if len(getattr(i, 'parent_ids', [])) == 1]
        if mutation_ideas:
            avg_mutation_fitness = sum(getattr(i, 'fitness', 0.0) for i in mutation_ideas) / len(mutation_ideas)
            mutation_op = OperationEffectiveness(
                operation_type="mutation",
                context=category,
                effectiveness_score=min(1.0, avg_mutation_fitness),
                usage_count=len(mutation_ideas),
                success_rate=len([i for i in mutation_ideas if getattr(i, 'fitness', 0.0) > 0.5]) / len(mutation_ideas),
                avg_fitness_improvement=avg_mutation_fitness
            )
            self.memory_db.store_operation_effectiveness(mutation_op)

    def _generate_insights(self, stats: dict[str, Any], category: str) -> list[str]:
        """Generate insights based on category statistics."""
        insights = []

        if stats.get("session_count", 0) == 0:
            insights.append(f"No data available for {category} category yet.")
            return insights

        session_count = stats["session_count"]
        avg_diversity = stats.get("avg_diversity", 0.0)
        avg_convergence = stats.get("avg_convergence", 0.0)
        avg_execution_time = stats.get("avg_execution_time", 0.0)

        insights.append(f"Based on {session_count} sessions in {category} category:")

        if avg_diversity > 0.7:
            insights.append("High diversity maintained - good exploration of solution space")
        elif avg_diversity < 0.3:
            insights.append("Low diversity observed - consider increasing mutation rate or population size")

        if avg_convergence < 3:
            insights.append("Fast convergence - might benefit from more generations or higher mutation")
        elif avg_convergence > 8:
            insights.append("Slow convergence - consider adjusting selection pressure or fitness weights")

        if avg_execution_time > 120:
            insights.append("Long execution times - consider reducing population size or generations")

        return insights


# Global memory system instance
_memory_system: MemorySystem | None = None


def get_memory_system() -> MemorySystem:
    """Get the global memory system instance."""
    global _memory_system
    if _memory_system is None:
        enable_learning = os.environ.get("GENETIC_MCP_MEMORY_ENABLED", "true").lower() == "true"
        db_path = os.environ.get("GENETIC_MCP_MEMORY_DB", "genetic_mcp_memory.db")
        _memory_system = MemorySystem(db_path, enable_learning)
    return _memory_system


def set_memory_system(memory_system: MemorySystem) -> None:
    """Set the global memory system instance."""
    global _memory_system
    _memory_system = memory_system
