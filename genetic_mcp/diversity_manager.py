"""Diversity preservation mechanisms for genetic algorithm."""

import logging
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .models import Idea

logger = logging.getLogger(__name__)


class DiversityManager:
    """Manages diversity preservation in genetic populations."""

    def __init__(
        self,
        min_distance_threshold: float = 0.3,
        niche_radius: float = 0.2,
        crowding_factor: int = 3
    ):
        self.min_distance_threshold = min_distance_threshold
        self.niche_radius = niche_radius
        self.crowding_factor = crowding_factor

        # Tracking
        self.species: dict[str, list[str]] = {}  # species_id -> idea_ids
        self.idea_species: dict[str, str] = {}  # idea_id -> species_id
        self.species_representatives: dict[str, np.ndarray] = {}  # species_id -> embedding

    def calculate_diversity_metrics(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Calculate various diversity metrics for the population."""
        if len(population) < 2:
            return {
                "simpson_diversity": 1.0,
                "shannon_diversity": 0.0,
                "average_distance": 0.0,
                "coverage": 1.0,
                "evenness": 1.0
            }

        # Get embeddings matrix
        embedding_matrix = np.array([
            embeddings.get(idea.id, np.zeros(768))
            for idea in population
        ])

        # Calculate pairwise distances
        similarity_matrix = cosine_similarity(embedding_matrix)
        distance_matrix = 1 - similarity_matrix

        # Average pairwise distance
        n = len(population)
        avg_distance = np.sum(distance_matrix) / (n * (n - 1))

        # Simpson's diversity index (probability two random ideas are different)
        unique_contents = defaultdict(int)
        for idea in population:
            # Use first 100 chars as signature
            signature = idea.content[:100]
            unique_contents[signature] += 1

        simpson = 1.0
        for count in unique_contents.values():
            simpson -= (count / n) ** 2

        # Shannon diversity index
        shannon = 0.0
        for count in unique_contents.values():
            if count > 0:
                p = count / n
                shannon -= p * np.log(p)

        # Coverage (how much of the idea space is covered)
        # Estimate using minimum spanning tree or convex hull volume
        coverage = self._estimate_coverage(embedding_matrix)

        # Evenness (how evenly distributed the population is)
        evenness = shannon / np.log(len(unique_contents)) if len(unique_contents) > 1 else 1.0

        return {
            "simpson_diversity": simpson,
            "shannon_diversity": shannon,
            "average_distance": avg_distance,
            "coverage": coverage,
            "evenness": evenness
        }

    def apply_crowding(
        self,
        population: list[Idea],
        offspring: list[Idea],
        embeddings: dict[str, np.ndarray]
    ) -> list[Idea]:
        """Apply deterministic crowding to maintain diversity."""
        new_population = []

        for child in offspring:
            if child.id not in embeddings:
                # If no embedding, add directly
                new_population.append(child)
                continue

            child_embedding = embeddings[child.id]

            # Find closest individuals in population
            distances = []
            for parent in population:
                if parent.id in embeddings:
                    parent_embedding = embeddings[parent.id]
                    distance = 1 - cosine_similarity(
                        child_embedding.reshape(1, -1),
                        parent_embedding.reshape(1, -1)
                    )[0, 0]
                    distances.append((distance, parent))

            if not distances:
                new_population.append(child)
                continue

            # Sort by distance
            distances.sort(key=lambda x: x[0])

            # Compare with closest individuals (crowding factor)
            replaced = False
            for i in range(min(self.crowding_factor, len(distances))):
                _, closest = distances[i]

                # Replace if child has better fitness
                if child.fitness > closest.fitness:
                    # Remove closest from population
                    population = [p for p in population if p.id != closest.id]
                    new_population.append(child)
                    replaced = True
                    break

            # If not replaced anyone, check if diverse enough
            if not replaced and distances[0][0] > self.min_distance_threshold:
                new_population.append(child)

        # Add remaining population members
        new_population.extend(population)

        return new_population

    def apply_speciation(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray]
    ) -> dict[str, list[Idea]]:
        """Divide population into species using clustering."""
        if len(population) < 3:
            # Too small for speciation
            return {"species_0": population}

        # Get embeddings matrix
        embedding_matrix = np.array([
            embeddings.get(idea.id, np.zeros(768))
            for idea in population
        ])

        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.niche_radius,
            min_samples=2,
            metric='cosine'
        ).fit(embedding_matrix)

        # Group by species
        species_dict = defaultdict(list)
        for idea, label in zip(population, clustering.labels_, strict=False):
            species_id = f"species_{label}" if label >= 0 else "species_outlier"
            species_dict[species_id].append(idea)
            self.idea_species[idea.id] = species_id

        # Update species tracking
        self.species = {
            species_id: [idea.id for idea in ideas]
            for species_id, ideas in species_dict.items()
        }

        # Calculate species representatives (centroids)
        for species_id, ideas in species_dict.items():
            species_embeddings = [
                embeddings[idea.id]
                for idea in ideas
                if idea.id in embeddings
            ]
            if species_embeddings:
                self.species_representatives[species_id] = np.mean(
                    species_embeddings, axis=0
                )

        return species_dict

    def calculate_crowding_distance(
        self,
        population: list[Idea],
        objectives: list[str] = None
    ) -> dict[str, float]:
        """Calculate crowding distance for each individual (NSGA-II style)."""
        if objectives is None:
            objectives = ["relevance", "novelty", "feasibility"]
        n = len(population)
        if n <= 2:
            return {idea.id: float('inf') for idea in population}

        # Initialize distances
        crowding_distances = {idea.id: 0.0 for idea in population}

        # For each objective
        for objective in objectives:
            # Sort by objective value
            sorted_pop = sorted(
                population,
                key=lambda x: x.scores.get(objective, 0)
            )

            # Boundary points get infinite distance
            crowding_distances[sorted_pop[0].id] = float('inf')
            crowding_distances[sorted_pop[-1].id] = float('inf')

            # Calculate distances for internal points
            obj_values = [idea.scores.get(objective, 0) for idea in sorted_pop]
            obj_range = max(obj_values) - min(obj_values)

            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                    crowding_distances[sorted_pop[i].id] += distance

        return crowding_distances

    def select_diverse_subset(
        self,
        population: list[Idea],
        subset_size: int,
        embeddings: dict[str, np.ndarray]
    ) -> list[Idea]:
        """Select diverse subset using maxmin algorithm."""
        if len(population) <= subset_size:
            return population

        selected = []
        remaining = population.copy()

        # Start with the best fitness individual
        best_idea = max(remaining, key=lambda x: x.fitness)
        selected.append(best_idea)
        remaining.remove(best_idea)

        # Iteratively select most distant individuals
        while len(selected) < subset_size and remaining:
            max_min_distance = -1
            most_distant = None

            for candidate in remaining:
                if candidate.id not in embeddings:
                    continue

                # Calculate minimum distance to selected set
                min_distance = float('inf')
                candidate_embedding = embeddings[candidate.id]

                for selected_idea in selected:
                    if selected_idea.id in embeddings:
                        selected_embedding = embeddings[selected_idea.id]
                        distance = 1 - cosine_similarity(
                            candidate_embedding.reshape(1, -1),
                            selected_embedding.reshape(1, -1)
                        )[0, 0]
                        min_distance = min(min_distance, distance)

                # Track candidate with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    most_distant = candidate

            if most_distant:
                selected.append(most_distant)
                remaining.remove(most_distant)
            else:
                # No valid candidates with embeddings
                break

        return selected

    def apply_fitness_sharing(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray],
        sharing_radius: float | None = None
    ) -> None:
        """Apply fitness sharing to promote diversity."""
        if not population:
            return

        sharing_radius = sharing_radius or self.niche_radius

        # Calculate similarity matrix
        n = len(population)
        sharing_matrix = np.zeros((n, n))

        for i in range(n):
            if population[i].id not in embeddings:
                continue

            embedding_i = embeddings[population[i].id]

            for j in range(i + 1, n):
                if population[j].id not in embeddings:
                    continue

                embedding_j = embeddings[population[j].id]

                # Calculate distance
                distance = 1 - cosine_similarity(
                    embedding_i.reshape(1, -1),
                    embedding_j.reshape(1, -1)
                )[0, 0]

                # Calculate sharing value
                if distance < sharing_radius:
                    sharing_value = 1 - (distance / sharing_radius)
                    sharing_matrix[i, j] = sharing_value
                    sharing_matrix[j, i] = sharing_value

        # Apply fitness sharing
        for i, idea in enumerate(population):
            # Calculate niche count
            niche_count = 1.0 + np.sum(sharing_matrix[i])

            # Adjust fitness
            idea.fitness = idea.fitness / niche_count
            idea.metadata["shared_fitness"] = idea.fitness
            idea.metadata["niche_count"] = niche_count

    def get_species_statistics(self) -> dict[str, dict[str, any]]:
        """Get statistics about current species."""
        stats = {}

        for species_id, idea_ids in self.species.items():
            stats[species_id] = {
                "size": len(idea_ids),
                "percentage": len(idea_ids) / sum(len(ids) for ids in self.species.values())
            }

            if species_id in self.species_representatives:
                stats[species_id]["has_representative"] = True

        stats["total_species"] = len(self.species)
        stats["singleton_species"] = sum(1 for ids in self.species.values() if len(ids) == 1)

        return stats

    def _estimate_coverage(self, embeddings: np.ndarray) -> float:
        """Estimate coverage of idea space."""
        if len(embeddings) < 2:
            return 0.0

        # Use principal components to estimate coverage
        from sklearn.decomposition import PCA

        try:
            # Project to 2D for volume estimation
            pca = PCA(n_components=min(2, len(embeddings) - 1))
            projected = pca.fit_transform(embeddings)

            # Calculate convex hull area (2D) as proxy for coverage
            from scipy.spatial import ConvexHull
            hull = ConvexHull(projected)

            # Normalize by number of points
            normalized_area = hull.volume / len(embeddings)

            # Scale to 0-1 range (empirically determined)
            coverage = min(1.0, normalized_area / 10.0)

            return coverage

        except Exception as e:
            logger.error(f"Coverage estimation failed: {e}")
            return 0.5

    def adaptive_niche_radius(
        self,
        population: list[Idea],
        target_species: int = 5
    ) -> float:
        """Adaptively adjust niche radius to maintain target number of species."""
        current_species = len(self.species)

        if current_species < target_species:
            # Too few species, decrease radius
            self.niche_radius *= 0.9
        elif current_species > target_species * 1.5:
            # Too many species, increase radius
            self.niche_radius *= 1.1

        # Clamp to reasonable range
        self.niche_radius = max(0.1, min(0.5, self.niche_radius))

        return self.niche_radius

    def get_diversity_preservation_actions(
        self,
        population: list[Idea],
        embeddings: dict[str, np.ndarray]
    ) -> list[str]:
        """Get recommended actions to preserve diversity."""
        actions = []

        metrics = self.calculate_diversity_metrics(population, embeddings)

        if metrics["simpson_diversity"] < 0.5:
            actions.append("Increase mutation rate - diversity is low")

        if metrics["average_distance"] < 0.2:
            actions.append("Apply fitness sharing - population is converging")

        if metrics["evenness"] < 0.6:
            actions.append("Use speciation - distribution is uneven")

        if metrics["coverage"] < 0.3:
            actions.append("Increase exploration - limited coverage of solution space")

        if len(self.species) < 3:
            actions.append("Reduce selection pressure - too few niches")

        return actions
