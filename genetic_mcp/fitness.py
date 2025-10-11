"""Fitness evaluation for ideas with multi-objective optimization support."""

import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .fitness_detail_aware import DetailMetrics
from .logging_config import log_operation, log_performance, setup_logging
from .models import FitnessWeights, Idea

logger = setup_logging(component="fitness")


class FitnessEvaluator:
    """Evaluates fitness of ideas based on multiple criteria."""

    def __init__(self, weights: FitnessWeights | None = None, use_detail_metrics: bool = False):
        """Initialize fitness evaluator.

        Args:
            weights: Fitness weights for base and detail metrics
            use_detail_metrics: Whether to use detail-aware metrics in feasibility calculation
        """
        self.weights = weights or FitnessWeights()
        self.embeddings_cache: dict[str, list[float]] = {}

        # Multi-objective optimization parameters
        self.use_pareto_ranking = False
        self.objective_names = ['relevance', 'novelty', 'feasibility']

        # Dynamic weight adjustment parameters
        self.weight_adaptation_rate = 0.1
        self.weight_history: list[FitnessWeights] = []

        # Detail-aware metrics
        self.use_detail_metrics = use_detail_metrics or self.weights.has_detail_weights()
        self.detail_metrics = DetailMetrics() if self.use_detail_metrics else None

    def calculate_fitness(self, idea: Idea, all_ideas: list[Idea],
                         target_embedding: list[float],
                         claude_evaluation_weight: float = 0.0) -> float:
        """Calculate overall fitness score for an idea.

        Mathematical formulation:
            fitness = w_r * relevance + w_n * novelty + w_f * feasibility

        where w_r, w_n, w_f are weights that sum to 1.0.

        If detail metrics are enabled, feasibility incorporates detail score:
            feasibility = (1 - β) * heuristic + β * detail_score
            detail_score = Σ(w_i * metric_i) for i in {δ, α, κ, τ}

        Args:
            idea: The idea to evaluate
            all_ideas: All ideas in the population (for novelty calculation)
            target_embedding: Target embedding for relevance calculation
            claude_evaluation_weight: Weight for Claude evaluation (0-1)

        Returns:
            Final fitness score in [0, 1]
        """
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

        # If Claude evaluation is available, combine scores
        if idea.claude_score is not None and claude_evaluation_weight > 0:
            algorithmic_weight = 1.0 - claude_evaluation_weight
            idea.combined_fitness = (
                algorithmic_weight * fitness +
                claude_evaluation_weight * idea.claude_score
            )
            logger.debug(f"Combined fitness for idea {idea.id}: "
                        f"algorithmic={fitness:.3f}, claude={idea.claude_score:.3f}, "
                        f"combined={idea.combined_fitness:.3f}")
            return idea.combined_fitness

        detail_info = ""
        if self.use_detail_metrics and 'detail_score' in idea.metadata:
            detail_info = f", detail={idea.metadata['detail_score']:.3f}"

        logger.debug(f"Calculated fitness for idea {idea.id}: "
                    f"fitness={fitness:.3f}, relevance={relevance:.3f}, "
                    f"novelty={novelty:.3f}, feasibility={feasibility:.3f}{detail_info}")

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
        """Calculate feasibility score.

        If detail metrics are enabled, combines heuristic feasibility with detail score.
        Otherwise, uses simple heuristic based on content length and structure.

        Mathematical formulation with detail metrics:
            feasibility = (1 - β) * heuristic_score + β * detail_score

        where β is the detail contribution weight (0.5 by default when detail metrics enabled).
        """
        # Calculate heuristic feasibility (baseline)
        heuristic_feasibility = self._calculate_heuristic_feasibility(idea)

        # If detail metrics not enabled, return heuristic score
        if not self.use_detail_metrics or self.detail_metrics is None:
            return heuristic_feasibility

        # Calculate detail score using configured weights
        detail_weights = {
            'delta': self.weights.implementation_depth,
            'alpha': self.weights.actionability,
            'kappa': self.weights.completeness,
            'tau': self.weights.technical_precision
        }

        # Normalize weights if they don't sum to 1
        weight_sum = sum(detail_weights.values())
        if weight_sum > 0:
            detail_weights = {k: v / weight_sum for k, v in detail_weights.items()}
        else:
            # Use equal weights if none specified
            detail_weights = {'delta': 0.25, 'alpha': 0.25, 'kappa': 0.25, 'tau': 0.25}

        detail_score = self.detail_metrics.calculate_detail_score(idea, detail_weights)

        # Store detail score in idea metadata
        idea.metadata['detail_score'] = detail_score
        idea.metadata['heuristic_feasibility'] = heuristic_feasibility

        # Combine heuristic and detail scores
        # β determines the contribution of detail metrics (default 0.5)
        beta = 0.5
        feasibility = (1 - beta) * heuristic_feasibility + beta * detail_score

        logger.debug(f"Feasibility for idea {idea.id}: {feasibility:.3f} "
                    f"(heuristic={heuristic_feasibility:.3f}, detail={detail_score:.3f})")

        return min(feasibility, 1.0)

    def _calculate_heuristic_feasibility(self, idea: Idea) -> float:
        """Calculate heuristic feasibility score based on content analysis.

        This is the baseline feasibility calculation without detail metrics.
        """
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

    def evaluate_population(self, ideas: list[Idea], target_embedding: list[float],
                          claude_evaluation_weight: float = 0.0) -> None:
        """Evaluate fitness for entire population."""
        start_time = time.time()
        log_operation(logger, "EVALUATE_POPULATION", population_size=len(ideas))

        for idea in ideas:
            self.calculate_fitness(idea, ideas, target_embedding, claude_evaluation_weight)

        # Log population statistics
        fitnesses = [idea.combined_fitness if idea.combined_fitness is not None else idea.fitness
                    for idea in ideas]
        log_performance(logger, "EVALUATE_POPULATION", time.time() - start_time,
                       population_size=len(ideas),
                       avg_fitness=np.mean(fitnesses),
                       max_fitness=max(fitnesses) if fitnesses else 0,
                       min_fitness=min(fitnesses) if fitnesses else 0)

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

    def get_selection_probabilities_shared(self, ideas: list[Idea], shared_fitness: dict[str, float]) -> list[float]:
        """Get selection probabilities based on shared fitness values."""
        if not ideas:
            return []

        # Use shared fitness values
        fitnesses = [shared_fitness.get(idea.id, idea.fitness) for idea in ideas]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return [1.0 / len(ideas)] * len(ideas)

        return [f / total_fitness for f in fitnesses]

    def tournament_select(self, ideas: list[Idea], tournament_size: int = 3) -> Idea:
        """Select idea using tournament selection."""
        if len(ideas) <= tournament_size:
            return max(ideas, key=lambda x: x.fitness)

        tournament = np.random.choice(ideas, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)

    def calculate_pareto_ranks(self, ideas: list[Idea]) -> dict[str, int]:
        """Calculate Pareto ranks for multi-objective optimization.

        Mathematical basis: Non-dominated sorting from NSGA-II.
        Rank 1 = Pareto optimal (non-dominated) solutions
        Rank 2 = Solutions dominated only by rank 1, etc.
        """
        n = len(ideas)
        if n == 0:
            return {}

        # Initialize dominance data structures
        domination_count = {idea.id: 0 for idea in ideas}  # Number of solutions dominating this one
        dominated_solutions = {idea.id: [] for idea in ideas}  # Solutions dominated by this one
        ranks = {}

        # Compare all pairs of solutions
        for i in range(n):
            for j in range(i + 1, n):
                idea_i, idea_j = ideas[i], ideas[j]

                # Check dominance relationship
                i_dominates_j = self._dominates(idea_i, idea_j)
                j_dominates_i = self._dominates(idea_j, idea_i)

                if i_dominates_j:
                    dominated_solutions[idea_i.id].append(idea_j.id)
                    domination_count[idea_j.id] += 1
                elif j_dominates_i:
                    dominated_solutions[idea_j.id].append(idea_i.id)
                    domination_count[idea_i.id] += 1

        # Assign ranks using non-dominated sorting
        current_rank = 1
        remaining_ideas = set(idea.id for idea in ideas)

        while remaining_ideas:
            # Find non-dominated solutions in current set
            current_front = []
            for idea_id in remaining_ideas:
                if domination_count[idea_id] == 0:
                    current_front.append(idea_id)
                    ranks[idea_id] = current_rank

            if not current_front:
                # Prevent infinite loop in case of errors
                for idea_id in remaining_ideas:
                    ranks[idea_id] = current_rank
                break

            # Remove current front from remaining
            for idea_id in current_front:
                remaining_ideas.remove(idea_id)
                # Reduce domination count for dominated solutions
                for dominated_id in dominated_solutions[idea_id]:
                    if dominated_id in domination_count:
                        domination_count[dominated_id] -= 1

            current_rank += 1

        return ranks

    def _dominates(self, idea1: Idea, idea2: Idea) -> bool:
        """Check if idea1 dominates idea2 in Pareto sense.

        idea1 dominates idea2 if:
        1. idea1 is no worse than idea2 in all objectives
        2. idea1 is strictly better than idea2 in at least one objective
        """
        better_in_one = False

        for objective in self.objective_names:
            score1 = idea1.scores.get(objective, 0)
            score2 = idea2.scores.get(objective, 0)

            if score1 < score2:
                return False  # idea1 is worse in this objective
            elif score1 > score2:
                better_in_one = True

        return better_in_one

    def evaluate_population_pareto(self, ideas: list[Idea], target_embedding: list[float]) -> None:
        """Evaluate population using Pareto ranking for multi-objective optimization."""
        # First calculate individual objective scores
        for idea in ideas:
            self.calculate_fitness(idea, ideas, target_embedding)

        # Calculate Pareto ranks
        pareto_ranks = self.calculate_pareto_ranks(ideas)

        # Assign fitness based on Pareto rank (lower rank = higher fitness)
        max_rank = max(pareto_ranks.values()) if pareto_ranks else 1

        for idea in ideas:
            rank = pareto_ranks.get(idea.id, max_rank)
            # Convert rank to fitness (inverse relationship)
            idea.fitness = 1.0 - (rank - 1) / max_rank
            idea.metadata['pareto_rank'] = rank

    def adapt_weights(self, population: list[Idea], target_objectives: dict[str, float] | None = None) -> None:
        """Dynamically adapt fitness weights based on population state.

        Mathematical basis: Gradient-based adaptation to balance objectives.
        If an objective is underperforming, increase its weight.
        """
        if not population:
            return

        # Calculate current objective statistics
        objective_stats = {obj: [] for obj in self.objective_names}

        for idea in population:
            for obj in self.objective_names:
                if obj in idea.scores:
                    objective_stats[obj].append(idea.scores[obj])

        # Calculate mean and variance for each objective
        obj_means = {}
        obj_vars = {}

        for obj, scores in objective_stats.items():
            if scores:
                obj_means[obj] = np.mean(scores)
                obj_vars[obj] = np.var(scores)
            else:
                obj_means[obj] = 0.5
                obj_vars[obj] = 0.1

        # Adapt weights based on objective performance
        new_weights = {
            'relevance': self.weights.relevance,
            'novelty': self.weights.novelty,
            'feasibility': self.weights.feasibility
        }

        # If target objectives provided, adapt towards them
        if target_objectives:
            for obj in self.objective_names:
                if obj in target_objectives:
                    current_mean = obj_means.get(obj, 0.5)
                    target = target_objectives[obj]

                    # Increase weight if below target
                    if current_mean < target:
                        new_weights[obj] *= (1 + self.weight_adaptation_rate)
                    elif current_mean > target * 1.2:  # Allow some overshoot
                        new_weights[obj] *= (1 - self.weight_adaptation_rate * 0.5)
        else:
            # Balance objectives based on variance (promote diversity)
            total_var = sum(obj_vars.values())
            if total_var > 0:
                for obj in self.objective_names:
                    # Increase weight for objectives with low variance
                    var_ratio = obj_vars[obj] / total_var
                    if var_ratio < 0.2:  # Low variance
                        new_weights[obj] *= (1 + self.weight_adaptation_rate * 0.5)

        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for obj in new_weights:
                new_weights[obj] /= total_weight

        # Apply smoothing to prevent drastic changes
        smoothing_factor = 0.7
        self.weights.relevance = smoothing_factor * self.weights.relevance + (1 - smoothing_factor) * new_weights['relevance']
        self.weights.novelty = smoothing_factor * self.weights.novelty + (1 - smoothing_factor) * new_weights['novelty']
        self.weights.feasibility = smoothing_factor * self.weights.feasibility + (1 - smoothing_factor) * new_weights['feasibility']

        # Store weight history
        self.weight_history.append(FitnessWeights(
            relevance=self.weights.relevance,
            novelty=self.weights.novelty,
            feasibility=self.weights.feasibility
        ))

    def calculate_hypervolume(self, ideas: list[Idea], reference_point: list[float] | None = None) -> float:
        """Calculate hypervolume indicator for multi-objective optimization.

        Mathematical basis: Volume of objective space dominated by the population.
        Higher hypervolume indicates better convergence and diversity.
        """
        if not ideas:
            return 0.0

        # Default reference point (worst possible values)
        if reference_point is None:
            reference_point = [0.0, 0.0, 0.0]  # For relevance, novelty, feasibility

        # Get objective values
        points = []
        for idea in ideas:
            point = []
            for obj in self.objective_names:
                point.append(idea.scores.get(obj, 0))
            points.append(point)

        if not points:
            return 0.0

        # Simple hypervolume approximation (exact calculation is complex)
        # This uses a Monte Carlo approximation
        n_samples = 10000
        n_objectives = len(self.objective_names)

        # Generate random points in the reference box
        count = 0
        for _ in range(n_samples):
            random_point = []
            for i in range(n_objectives):
                random_point.append(np.random.uniform(reference_point[i], 1.0))

            # Check if dominated by any solution
            dominated = False
            for point in points:
                if all(point[i] >= random_point[i] for i in range(n_objectives)):
                    dominated = True
                    break

            if dominated:
                count += 1

        # Calculate volume
        reference_volume = np.prod([1.0 - ref for ref in reference_point])
        hypervolume = (count / n_samples) * reference_volume

        return hypervolume
