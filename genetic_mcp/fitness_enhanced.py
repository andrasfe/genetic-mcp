"""Enhanced fitness evaluation with multi-objective optimization and LLM-based scoring."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .llm_client import LLMClient
from .models import FitnessWeights, Idea

logger = logging.getLogger(__name__)


@dataclass
class FitnessComponents:
    """Detailed fitness components for analysis."""
    relevance: float = 0.0
    novelty: float = 0.0
    feasibility: float = 0.0
    coherence: float = 0.0
    innovation: float = 0.0
    practicality: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "novelty": self.novelty,
            "feasibility": self.feasibility,
            "coherence": self.coherence,
            "innovation": self.innovation,
            "practicality": self.practicality
        }


class EnhancedFitnessEvaluator:
    """Enhanced fitness evaluator with LLM-based scoring and multi-objective optimization."""

    def __init__(
        self,
        weights: FitnessWeights | None = None,
        llm_client: LLMClient | None = None,
        use_pareto: bool = True
    ):
        self.weights = weights or FitnessWeights()
        self.llm_client = llm_client
        self.use_pareto = use_pareto

        # Caches
        self.embeddings_cache: dict[str, np.ndarray] = {}
        self.llm_scores_cache: dict[str, FitnessComponents] = {}
        self.similarity_matrix_cache: np.ndarray | None = None

        # Dynamic weight adjustment
        self.dynamic_weights = self.weights.model_copy()
        self.weight_history: list[FitnessWeights] = []

    async def evaluate_population_enhanced(
        self,
        ideas: list[Idea],
        target_embedding: np.ndarray,
        target_prompt: str,
        generation: int = 0
    ) -> None:
        """Evaluate entire population with enhanced metrics."""
        # Pre-compute embeddings for all ideas
        await self._compute_all_embeddings(ideas)

        # Pre-compute similarity matrix for novelty calculation
        self._compute_similarity_matrix(ideas)

        # Batch LLM evaluations for efficiency
        if self.llm_client:
            await self._batch_llm_evaluation(ideas, target_prompt)

        # Calculate fitness for each idea
        for idea in ideas:
            components = await self._calculate_all_components(
                idea, ideas, target_embedding, target_prompt
            )

            # Store detailed scores
            idea.scores = components.to_dict()

            # Calculate weighted fitness
            if self.use_pareto:
                # For Pareto optimization, store components separately
                idea.metadata["fitness_components"] = components
                idea.fitness = self._calculate_weighted_fitness(components)
            else:
                idea.fitness = self._calculate_weighted_fitness(components)

        # Apply Pareto ranking if enabled
        if self.use_pareto:
            self._apply_pareto_ranking(ideas)

        # Update dynamic weights based on generation
        self._update_dynamic_weights(ideas, generation)

    async def _calculate_all_components(
        self,
        idea: Idea,
        all_ideas: list[Idea],
        target_embedding: np.ndarray,
        target_prompt: str
    ) -> FitnessComponents:
        """Calculate all fitness components for an idea."""
        components = FitnessComponents()

        # Relevance (semantic similarity to target)
        components.relevance = self._calculate_enhanced_relevance(
            idea, target_embedding
        )

        # Novelty (diversity from other ideas)
        components.novelty = self._calculate_enhanced_novelty(
            idea, all_ideas
        )

        # LLM-based scores if available
        if idea.id in self.llm_scores_cache:
            llm_components = self.llm_scores_cache[idea.id]
            components.feasibility = llm_components.feasibility
            components.coherence = llm_components.coherence
            components.innovation = llm_components.innovation
            components.practicality = llm_components.practicality
        else:
            # Fallback to heuristic scores
            components.feasibility = self._heuristic_feasibility(idea)
            components.coherence = self._heuristic_coherence(idea)
            components.innovation = components.novelty * 0.8
            components.practicality = components.feasibility * 0.9

        return components

    def _calculate_enhanced_relevance(
        self,
        idea: Idea,
        target_embedding: np.ndarray
    ) -> float:
        """Enhanced relevance calculation with multiple factors."""
        if idea.id not in self.embeddings_cache:
            return 0.5

        idea_embedding = self.embeddings_cache[idea.id]

        # Cosine similarity
        cosine_sim = cosine_similarity(
            idea_embedding.reshape(1, -1),
            target_embedding.reshape(1, -1)
        )[0, 0]

        # Normalize to 0-1 range
        base_relevance = (cosine_sim + 1) / 2

        # Adjust based on content length (penalize too short/long)
        length_factor = self._length_penalty(len(idea.content.split()))

        return base_relevance * length_factor

    def _calculate_enhanced_novelty(
        self,
        idea: Idea,
        all_ideas: list[Idea]
    ) -> float:
        """Enhanced novelty using pre-computed similarity matrix."""
        if idea.id not in self.embeddings_cache or len(all_ideas) <= 1:
            return 1.0

        # Get idea index
        idea_idx = next(
            (i for i, other in enumerate(all_ideas) if other.id == idea.id),
            None
        )

        if idea_idx is None or self.similarity_matrix_cache is None:
            return 0.5

        # Calculate average distance to other ideas
        similarities = self.similarity_matrix_cache[idea_idx]
        # Exclude self-similarity
        other_similarities = np.concatenate([
            similarities[:idea_idx],
            similarities[idea_idx + 1:]
        ])

        if len(other_similarities) == 0:
            return 1.0

        # Novelty is inverse of average similarity
        avg_similarity = np.mean(other_similarities)
        novelty = 1.0 - avg_similarity

        # Boost novelty for ideas that are very different from any other
        min_similarity = np.min(other_similarities)
        if min_similarity < 0.3:
            novelty = min(1.0, novelty * 1.2)

        return novelty

    async def _batch_llm_evaluation(
        self,
        ideas: list[Idea],
        target_prompt: str
    ) -> None:
        """Batch evaluate ideas using LLM for efficiency."""
        if not self.llm_client:
            return

        # Group ideas for batch processing
        batch_size = 5
        for i in range(0, len(ideas), batch_size):
            batch = ideas[i:i + batch_size]

            # Create batch evaluation prompt
            eval_prompt = self._create_batch_eval_prompt(batch, target_prompt)

            try:
                response = await self.llm_client.generate(
                    eval_prompt,
                    temperature=0.3,  # Low temperature for consistent scoring
                    max_tokens=1000
                )

                # Parse and cache scores
                self._parse_batch_scores(batch, response)

            except Exception as e:
                logger.error(f"Batch LLM evaluation failed: {e}")

    def _create_batch_eval_prompt(
        self,
        ideas: list[Idea],
        target_prompt: str
    ) -> str:
        """Create prompt for batch evaluation."""
        prompt = f"""
        Evaluate the following ideas for the prompt: "{target_prompt}"

        Score each idea on these criteria (0-10):
        - Feasibility: How practical and implementable is this idea?
        - Coherence: How well-structured and clear is the idea?
        - Innovation: How creative and novel is the approach?
        - Practicality: How useful would this be in real-world applications?

        Ideas to evaluate:
        """

        for i, idea in enumerate(ideas):
            prompt += f"\n\nIDEA {i+1}:\n{idea.content}"

        prompt += """

        Provide scores in this format for each idea:
        IDEA 1: feasibility=X, coherence=Y, innovation=Z, practicality=W
        IDEA 2: feasibility=X, coherence=Y, innovation=Z, practicality=W
        ...
        """

        return prompt

    def _parse_batch_scores(self, ideas: list[Idea], response: str) -> None:
        """Parse LLM response and cache scores."""
        lines = response.strip().split('\n')

        for line in lines:
            if line.startswith("IDEA"):
                try:
                    # Extract idea number and scores
                    parts = line.split(':')
                    if len(parts) != 2:
                        continue

                    idea_num = int(parts[0].replace("IDEA", "").strip()) - 1
                    if 0 <= idea_num < len(ideas):
                        scores_str = parts[1].strip()

                        # Parse scores
                        components = FitnessComponents()
                        for score_part in scores_str.split(','):
                            if '=' in score_part:
                                key, value = score_part.split('=')
                                key = key.strip()
                                value = float(value.strip()) / 10.0  # Normalize to 0-1

                                if hasattr(components, key):
                                    setattr(components, key, value)

                        # Cache the scores
                        self.llm_scores_cache[ideas[idea_num].id] = components

                except Exception as e:
                    logger.error(f"Failed to parse score line: {line}, error: {e}")

    def _apply_pareto_ranking(self, ideas: list[Idea]) -> None:
        """Apply Pareto ranking for multi-objective optimization."""
        n = len(ideas)
        if n == 0:
            return

        # Extract objective values
        objectives = np.array([
            [
                idea.scores.get("relevance", 0),
                idea.scores.get("novelty", 0),
                idea.scores.get("feasibility", 0),
                idea.scores.get("innovation", 0)
            ]
            for idea in ideas
        ])

        # Calculate domination counts
        domination_counts = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(objectives[i] >= objectives[j]) and np.any(objectives[i] > objectives[j]):
                    domination_counts[j] += 1
                    dominated_by[i].append(j)
                # Check if j dominates i
                elif np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    domination_counts[i] += 1
                    dominated_by[j].append(i)

        # Assign Pareto ranks
        ranks = np.zeros(n, dtype=int)
        current_rank = 1

        while np.any(domination_counts >= 0):
            # Find non-dominated solutions
            current_front = np.where(domination_counts == 0)[0]

            if len(current_front) == 0:
                break

            # Assign rank
            for idx in current_front:
                ranks[idx] = current_rank

                # Update domination counts
                for dominated_idx in dominated_by[idx]:
                    domination_counts[dominated_idx] -= 1

                domination_counts[idx] = -1  # Mark as processed

            current_rank += 1

        # Update fitness based on Pareto rank and crowding distance
        for i, idea in enumerate(ideas):
            # Lower rank is better
            pareto_fitness = 1.0 / ranks[i] if ranks[i] > 0 else 1.0

            # Combine with weighted fitness
            idea.fitness = 0.7 * idea.fitness + 0.3 * pareto_fitness
            idea.metadata["pareto_rank"] = int(ranks[i])

    def _calculate_weighted_fitness(self, components: FitnessComponents) -> float:
        """Calculate weighted fitness from components."""
        # Use dynamic weights if available
        weights = self.dynamic_weights

        # Extended fitness calculation including new components
        fitness = (
            weights.relevance * components.relevance +
            weights.novelty * components.novelty +
            weights.feasibility * components.feasibility +
            0.1 * components.coherence +  # Additional weight for coherence
            0.1 * components.innovation +  # Additional weight for innovation
            0.1 * components.practicality  # Additional weight for practicality
        )

        # Normalize to ensure fitness is in [0, 1]
        return min(1.0, fitness / 1.3)  # Divide by sum of all weights

    def _update_dynamic_weights(self, ideas: list[Idea], generation: int) -> None:
        """Dynamically adjust weights based on population characteristics."""
        if not ideas:
            return

        # Calculate population statistics
        relevances = [idea.scores.get("relevance", 0) for idea in ideas]
        novelties = [idea.scores.get("novelty", 0) for idea in ideas]
        feasibilities = [idea.scores.get("feasibility", 0) for idea in ideas]

        avg_relevance = np.mean(relevances)
        avg_novelty = np.mean(novelties)
        np.mean(feasibilities)

        # Adjust weights to balance objectives
        # If relevance is low, increase its weight
        if avg_relevance < 0.4:
            self.dynamic_weights.relevance = min(0.6, self.weights.relevance * 1.2)
        elif avg_relevance > 0.8:
            self.dynamic_weights.relevance = max(0.2, self.weights.relevance * 0.8)

        # If novelty is low (converging), increase its weight
        if avg_novelty < 0.3:
            self.dynamic_weights.novelty = min(0.5, self.weights.novelty * 1.3)

        # Early generations: focus on exploration (novelty)
        # Later generations: focus on exploitation (feasibility)
        if generation < 3:
            self.dynamic_weights.novelty = min(0.5, self.dynamic_weights.novelty * 1.1)
            self.dynamic_weights.feasibility = max(0.2, self.dynamic_weights.feasibility * 0.9)
        else:
            self.dynamic_weights.novelty = max(0.2, self.dynamic_weights.novelty * 0.95)
            self.dynamic_weights.feasibility = min(0.5, self.dynamic_weights.feasibility * 1.05)

        # Normalize weights to sum to 1
        total = (
            self.dynamic_weights.relevance +
            self.dynamic_weights.novelty +
            self.dynamic_weights.feasibility
        )
        self.dynamic_weights.relevance /= total
        self.dynamic_weights.novelty /= total
        self.dynamic_weights.feasibility /= total

        # Store weight history
        self.weight_history.append(self.dynamic_weights.model_copy())

    async def _compute_all_embeddings(self, ideas: list[Idea]) -> None:
        """Compute embeddings for all ideas."""
        # This would typically use the LLM client to generate embeddings
        # For now, using placeholder
        for idea in ideas:
            if idea.id not in self.embeddings_cache:
                # In real implementation, this would call llm_client.embed()
                self.embeddings_cache[idea.id] = np.random.randn(768)  # Placeholder

    def _compute_similarity_matrix(self, ideas: list[Idea]) -> None:
        """Pre-compute similarity matrix for efficiency."""
        len(ideas)
        embeddings = []

        for idea in ideas:
            if idea.id in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[idea.id])
            else:
                embeddings.append(np.zeros(768))  # Placeholder

        embeddings_matrix = np.array(embeddings)

        # Compute pairwise cosine similarities
        self.similarity_matrix_cache = cosine_similarity(embeddings_matrix)

    def _length_penalty(self, word_count: int) -> float:
        """Calculate length penalty factor."""
        if word_count < 20:
            return word_count / 20
        elif word_count > 200:
            return max(0.5, 1.0 - (word_count - 200) / 400)
        else:
            return 1.0

    def _heuristic_feasibility(self, idea: Idea) -> float:
        """Fallback heuristic feasibility score."""
        content = idea.content.lower()
        score = 0.5

        # Positive indicators
        if any(word in content for word in ["implement", "build", "create", "develop"]):
            score += 0.1
        if any(word in content for word in ["step", "phase", "process", "method"]):
            score += 0.1
        if len(content.split('.')) > 2:  # Multiple sentences
            score += 0.1

        # Negative indicators
        if any(word in content for word in ["maybe", "possibly", "might", "could"]):
            score -= 0.1
        if len(content) < 50:
            score -= 0.2

        return max(0, min(1, score))

    def _heuristic_coherence(self, idea: Idea) -> float:
        """Fallback heuristic coherence score."""
        sentences = idea.content.split('.')

        # Base score
        score = 0.6

        # Check for structure
        if len(sentences) >= 2:
            score += 0.1
        if any(marker in idea.content for marker in ["1.", "2.", "-", "*"]):
            score += 0.1
        if len(idea.content.split()) > 30:
            score += 0.1

        # Check for transition words
        transitions = ["furthermore", "additionally", "however", "therefore", "thus"]
        if any(word in idea.content.lower() for word in transitions):
            score += 0.1

        return min(1.0, score)

    def get_fitness_statistics(self, ideas: list[Idea]) -> dict[str, dict[str, float]]:
        """Get detailed fitness statistics for analysis."""
        if not ideas:
            return {}

        stats = {
            "overall": {
                "mean": np.mean([idea.fitness for idea in ideas]),
                "std": np.std([idea.fitness for idea in ideas]),
                "max": max(idea.fitness for idea in ideas),
                "min": min(idea.fitness for idea in ideas)
            }
        }

        # Component statistics
        for component in ["relevance", "novelty", "feasibility", "coherence", "innovation", "practicality"]:
            values = [idea.scores.get(component, 0) for idea in ideas if component in idea.scores]
            if values:
                stats[component] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "max": max(values),
                    "min": min(values)
                }

        return stats
