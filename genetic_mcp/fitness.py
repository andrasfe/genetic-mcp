"""Fitness evaluation for ideas."""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import FitnessWeights, Idea

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Evaluates fitness of ideas based on multiple criteria."""

    def __init__(self, weights: FitnessWeights | None = None):
        self.weights = weights or FitnessWeights()
        self.embeddings_cache: dict[str, list[float]] = {}

    def calculate_fitness(self, idea: Idea, all_ideas: list[Idea],
                         target_embedding: list[float]) -> float:
        """Calculate overall fitness score for an idea."""
        # Get individual scores
        relevance = self._calculate_relevance(idea, target_embedding)
        novelty = self._calculate_novelty(idea, all_ideas)
        feasibility = self._calculate_feasibility(idea)

        # Store scores
        idea.scores = {
            "relevance": relevance,
            "novelty": novelty,
            "feasibility": feasibility
        }

        # Calculate weighted fitness
        fitness = (
            self.weights.relevance * relevance +
            self.weights.novelty * novelty +
            self.weights.feasibility * feasibility
        )

        idea.fitness = fitness
        return fitness

    def _calculate_relevance(self, idea: Idea, target_embedding: list[float]) -> float:
        """Calculate relevance score using cosine similarity."""
        if idea.id not in self.embeddings_cache:
            logger.warning(f"No embedding for idea {idea.id}, using default relevance")
            return 0.5

        idea_embedding = self.embeddings_cache[idea.id]
        similarity = cosine_similarity(
            np.array(idea_embedding).reshape(1, -1),
            np.array(target_embedding).reshape(1, -1)
        )[0, 0]

        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1) / 2

    def _calculate_novelty(self, idea: Idea, all_ideas: list[Idea]) -> float:
        """Calculate novelty score based on distance from other ideas."""
        if idea.id not in self.embeddings_cache:
            return 0.5

        if len(all_ideas) <= 1:
            return 1.0  # First idea is maximally novel

        idea_embedding = np.array(self.embeddings_cache[idea.id])

        # Calculate distances to all other ideas
        distances = []
        for other in all_ideas:
            if other.id == idea.id or other.id not in self.embeddings_cache:
                continue

            other_embedding = np.array(self.embeddings_cache[other.id])
            distance = np.linalg.norm(idea_embedding - other_embedding)
            distances.append(distance)

        if not distances:
            return 0.5

        # Normalize by maximum possible distance (roughly sqrt(embedding_dim))
        embedding_dim = len(idea_embedding)
        max_distance = np.sqrt(embedding_dim) * 2  # Rough estimate

        avg_distance = np.mean(distances)
        novelty = min(avg_distance / max_distance, 1.0)

        return novelty

    def _calculate_feasibility(self, idea: Idea) -> float:
        """Calculate feasibility score (placeholder for critic model)."""
        # In a real implementation, this would use a critic model
        # For now, we'll use a simple heuristic based on content length
        # and structure

        content = idea.content.lower()

        # Penalize very short or very long ideas
        length_score = 1.0
        word_count = len(content.split())
        if word_count < 10:
            length_score = word_count / 10
        elif word_count > 500:
            length_score = max(0.3, 1.0 - (word_count - 500) / 1000)

        # Bonus for structured content
        structure_score = 0.7  # Base score
        if any(marker in content for marker in ["1.", "2.", "•", "-", "*"]):
            structure_score += 0.15
        if any(word in content for word in ["implementation", "approach", "method", "solution"]):
            structure_score += 0.15

        # Combine scores
        feasibility = (length_score + structure_score) / 2
        return min(feasibility, 1.0)

    def add_embedding(self, idea_id: str, embedding: list[float]) -> None:
        """Add embedding to cache."""
        self.embeddings_cache[idea_id] = embedding

    def evaluate_population(self, ideas: list[Idea], target_embedding: list[float]) -> None:
        """Evaluate fitness for entire population."""
        for idea in ideas:
            self.calculate_fitness(idea, ideas, target_embedding)

    def get_selection_probabilities(self, ideas: list[Idea]) -> list[float]:
        """Get selection probabilities based on fitness (roulette wheel)."""
        if not ideas:
            return []

        fitnesses = [idea.fitness for idea in ideas]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            # Equal probability if all fitness is 0
            return [1.0 / len(ideas)] * len(ideas)

        return [f / total_fitness for f in fitnesses]

    def tournament_select(self, ideas: list[Idea], tournament_size: int = 3) -> Idea:
        """Select idea using tournament selection."""
        if len(ideas) <= tournament_size:
            return max(ideas, key=lambda x: x.fitness)

        tournament = np.random.choice(ideas, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
