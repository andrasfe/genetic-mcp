"""GPU-optimized diversity metrics and population analysis for genetic algorithms.

This module provides GPU-accelerated implementations of diversity metrics including
phenotypic/genotypic diversity, entropy measures, and population statistics.
"""

import logging

import numpy as np

from .gpu_accelerated import GPUConfig, GPUMemoryManager

# GPU libraries with fallback imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU diversity metrics disabled")

CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUDiversityMetrics:
    """GPU-optimized diversity metrics for genetic algorithm populations."""

    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.memory_manager = GPUMemoryManager(self.config)
        self.use_gpu = self.memory_manager.is_gpu_available()
        self.device = torch.device(self.config.device) if TORCH_AVAILABLE else None

        logger.info(f"Initialized GPU diversity metrics on {self.config.device}")

    def calculate_population_diversity_gpu(
        self,
        embeddings: np.ndarray,
        fitness_scores: np.ndarray | None = None,
        objective_values: np.ndarray | None = None
    ) -> dict[str, float]:
        """Calculate comprehensive diversity metrics on GPU.

        Returns:
            Dictionary containing various diversity metrics:
            - embedding_diversity: Average pairwise distance in embedding space
            - fitness_diversity: Variance in fitness scores
            - coverage: Hypervolume coverage in objective space
            - entropy: Population entropy based on clustering
            - uniqueness: Ratio of unique solutions
        """
        metrics = {}

        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._calculate_diversity_cpu(embeddings, fitness_scores, objective_values)

        # Convert to GPU tensors
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        n_individuals = embeddings_tensor.shape[0]

        # 1. Embedding space diversity
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # Efficient pairwise distance calculation
            distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)

            # Get upper triangle (excluding diagonal)
            mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
            pairwise_distances = distances[mask]

            metrics['embedding_diversity'] = float(pairwise_distances.mean())
            metrics['embedding_diversity_std'] = float(pairwise_distances.std())
            metrics['min_pairwise_distance'] = float(pairwise_distances.min())
            metrics['max_pairwise_distance'] = float(pairwise_distances.max())

        # 2. Fitness diversity
        if fitness_scores is not None:
            fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)
            metrics['fitness_mean'] = float(fitness_tensor.mean())
            metrics['fitness_std'] = float(fitness_tensor.std())
            metrics['fitness_variance'] = float(fitness_tensor.var())
            metrics['fitness_range'] = float(fitness_tensor.max() - fitness_tensor.min())

            # Fitness entropy
            if fitness_tensor.max() > fitness_tensor.min():
                # Discretize fitness into bins for entropy calculation
                n_bins = min(20, n_individuals // 2)
                hist = torch.histc(fitness_tensor, bins=n_bins)
                hist = hist[hist > 0]  # Remove empty bins
                probs = hist / hist.sum()
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                metrics['fitness_entropy'] = float(entropy)

        # 3. Objective space diversity
        if objective_values is not None:
            obj_tensor = torch.from_numpy(objective_values).float().to(self.device)

            # Calculate hypervolume approximation
            hypervolume = self._calculate_hypervolume_gpu(obj_tensor)
            metrics['hypervolume'] = float(hypervolume)

            # Calculate spread in each objective
            for i in range(obj_tensor.shape[1]):
                metrics[f'objective_{i}_spread'] = float(obj_tensor[:, i].std())

        # 4. Clustering-based diversity
        cluster_metrics = self._calculate_cluster_diversity_gpu(embeddings_tensor)
        metrics.update(cluster_metrics)

        # 5. Uniqueness ratio
        # Check for near-duplicate solutions
        duplicate_threshold = 1e-3
        unique_mask = self._find_unique_solutions_gpu(embeddings_tensor, duplicate_threshold)
        metrics['uniqueness_ratio'] = float(unique_mask.sum()) / n_individuals

        return metrics

    def _calculate_hypervolume_gpu(
        self,
        objectives: "torch.Tensor",
        reference_point: "torch.Tensor | None" = None,
        n_samples: int = 10000
    ) -> float:
        """GPU-accelerated hypervolume calculation using Monte Carlo approximation."""
        n_points, n_objectives = objectives.shape

        # Default reference point (worst case)
        if reference_point is None:
            reference_point = torch.zeros(n_objectives, device=self.device)
        else:
            reference_point = reference_point.to(self.device)

        # Normalize objectives to [0, 1] range
        obj_min = objectives.min(dim=0)[0]
        obj_max = objectives.max(dim=0)[0]
        obj_range = obj_max - obj_min + 1e-6

        objectives_norm = (objectives - obj_min) / obj_range
        reference_norm = (reference_point - obj_min) / obj_range

        # Generate random points in the reference box
        random_points = torch.rand(n_samples, n_objectives, device=self.device)

        # Scale random points to the reference box
        for i in range(n_objectives):
            random_points[:, i] = random_points[:, i] * (1.0 - reference_norm[i]) + reference_norm[i]

        # Check dominance using broadcasting
        # A point is dominated if any solution dominates it in all objectives
        dominated_count = 0

        # Process in batches to manage memory
        batch_size = min(1000, n_samples)
        for i in range(0, n_samples, batch_size):
            batch_points = random_points[i:i+batch_size]

            # Check if batch points are dominated by any solution
            # Broadcasting: (batch_size, 1, n_objectives) >= (1, n_points, n_objectives)
            dominance_check = (batch_points.unsqueeze(1) <= objectives_norm.unsqueeze(0)).all(dim=2)
            dominated = dominance_check.any(dim=1)
            dominated_count += dominated.sum().item()

        # Calculate hypervolume
        reference_volume = torch.prod(1.0 - reference_norm).item()
        hypervolume = (dominated_count / n_samples) * reference_volume

        return hypervolume

    def _calculate_cluster_diversity_gpu(
        self,
        embeddings: "torch.Tensor",
        n_clusters: int | None = None
    ) -> dict[str, float]:
        """Calculate diversity metrics based on clustering."""
        n_points = embeddings.shape[0]

        if n_clusters is None:
            n_clusters = min(int(np.sqrt(n_points)), 10)

        # Simple K-means clustering on GPU
        centroids, labels = self._kmeans_gpu(embeddings, n_clusters, max_iters=30)

        # Calculate cluster statistics
        cluster_sizes = torch.bincount(labels, minlength=n_clusters)

        # Remove empty clusters
        non_empty = cluster_sizes > 0
        cluster_sizes = cluster_sizes[non_empty].float()

        # Calculate entropy of cluster distribution
        cluster_probs = cluster_sizes / cluster_sizes.sum()
        cluster_entropy = -torch.sum(cluster_probs * torch.log2(cluster_probs + 1e-10))

        # Calculate average intra-cluster distance
        intra_distances = []
        for i in range(n_clusters):
            if cluster_sizes[i] > 1:
                cluster_points = embeddings[labels == i]
                cluster_dists = torch.cdist(cluster_points, cluster_points, p=2)
                mask = torch.triu(torch.ones_like(cluster_dists, dtype=torch.bool), diagonal=1)
                if mask.any():
                    intra_distances.append(cluster_dists[mask].mean())

        avg_intra_distance = torch.stack(intra_distances).mean() if intra_distances else torch.tensor(0.0)

        # Calculate inter-cluster distance
        if len(centroids) > 1:
            inter_distances = torch.cdist(centroids[non_empty], centroids[non_empty], p=2)
            mask = torch.triu(torch.ones_like(inter_distances, dtype=torch.bool), diagonal=1)
            avg_inter_distance = inter_distances[mask].mean() if mask.any() else torch.tensor(0.0)
        else:
            avg_inter_distance = torch.tensor(0.0)

        return {
            'cluster_entropy': float(cluster_entropy),
            'n_effective_clusters': int(non_empty.sum()),
            'avg_intra_cluster_distance': float(avg_intra_distance),
            'avg_inter_cluster_distance': float(avg_inter_distance),
            'cluster_separation': float(avg_inter_distance / (avg_intra_distance + 1e-6))
        }

    def _kmeans_gpu(
        self,
        data: "torch.Tensor",
        n_clusters: int,
        max_iters: int = 30
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Simple K-means implementation on GPU."""
        data.shape[0]

        # Initialize centroids using K-means++
        centroids = self._kmeans_plusplus_init_gpu(data, n_clusters)

        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = torch.cdist(data, centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]

            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        return centroids, labels

    def _kmeans_plusplus_init_gpu(
        self,
        data: "torch.Tensor",
        n_clusters: int
    ) -> "torch.Tensor":
        """K-means++ initialization on GPU."""
        n_points = data.shape[0]
        centroids = []

        # Choose first centroid randomly
        first_idx = torch.randint(n_points, (1,), device=self.device).item()
        centroids.append(data[first_idx])

        # Choose remaining centroids
        for _ in range(1, n_clusters):
            # Calculate distances to nearest centroid
            centroid_tensor = torch.stack(centroids)
            distances = torch.cdist(data, centroid_tensor, p=2)
            min_distances = distances.min(dim=1)[0]

            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()

            # Sample next centroid
            next_idx = torch.multinomial(probabilities, 1).item()
            centroids.append(data[next_idx])

        return torch.stack(centroids)

    def _find_unique_solutions_gpu(
        self,
        embeddings: "torch.Tensor",
        threshold: float
    ) -> "torch.Tensor":
        """Find unique solutions based on embedding distance threshold."""
        n_points = embeddings.shape[0]
        unique_mask = torch.ones(n_points, dtype=torch.bool, device=self.device)

        # Calculate pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Mark duplicates (keep first occurrence)
        for i in range(n_points):
            if unique_mask[i]:
                # Find all points too close to this one
                close_points = (distances[i] < threshold) & (torch.arange(n_points, device=self.device) > i)
                unique_mask[close_points] = False

        return unique_mask

    def calculate_convergence_metrics_gpu(
        self,
        population_history: list[np.ndarray],
        fitness_history: list[np.ndarray]
    ) -> dict[str, float]:
        """Calculate convergence metrics over multiple generations."""
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._calculate_convergence_cpu(population_history, fitness_history)

        metrics = {}

        # Convert history to tensors
        fitness_tensors = [torch.from_numpy(f).float().to(self.device) for f in fitness_history]

        # Fitness improvement rate
        if len(fitness_tensors) > 1:
            best_fitness_history = torch.stack([f.max() for f in fitness_tensors])
            avg_fitness_history = torch.stack([f.mean() for f in fitness_tensors])

            # Calculate improvement rates
            best_improvements = best_fitness_history[1:] - best_fitness_history[:-1]
            avg_improvements = avg_fitness_history[1:] - avg_fitness_history[:-1]

            metrics['best_fitness_improvement_rate'] = float(best_improvements.mean())
            metrics['avg_fitness_improvement_rate'] = float(avg_improvements.mean())
            metrics['fitness_stagnation_ratio'] = float((best_improvements <= 1e-6).float().mean())

        # Population diversity over time
        if len(population_history) > 1:
            diversity_history = []
            for pop_embeddings in population_history:
                pop_tensor = torch.from_numpy(pop_embeddings).float().to(self.device)
                distances = torch.cdist(pop_tensor, pop_tensor, p=2)
                mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
                avg_distance = distances[mask].mean()
                diversity_history.append(avg_distance)

            diversity_tensor = torch.stack(diversity_history)
            metrics['diversity_trend'] = float((diversity_tensor[-1] - diversity_tensor[0]) / len(diversity_history))
            metrics['final_diversity'] = float(diversity_tensor[-1])

        return metrics

    def batch_diversity_calculation_gpu(
        self,
        populations: list[np.ndarray],
        fitness_scores_list: list[np.ndarray],
        batch_size: int = 10
    ) -> list[dict[str, float]]:
        """Calculate diversity metrics for multiple populations in batch."""
        if not self.use_gpu or not TORCH_AVAILABLE:
            return [self._calculate_diversity_cpu(pop, fit, None)
                   for pop, fit in zip(populations, fitness_scores_list, strict=False)]

        results = []

        # Process in batches
        for i in range(0, len(populations), batch_size):
            batch_populations = populations[i:i+batch_size]
            batch_fitness = fitness_scores_list[i:i+batch_size]

            # Calculate metrics for each population in batch
            batch_results = []
            for pop, fit in zip(batch_populations, batch_fitness, strict=False):
                metrics = self.calculate_population_diversity_gpu(pop, fit)
                batch_results.append(metrics)

            results.extend(batch_results)

        return results

    # CPU fallback methods
    def _calculate_diversity_cpu(
        self,
        embeddings: np.ndarray,
        fitness_scores: np.ndarray | None = None,
        objective_values: np.ndarray | None = None
    ) -> dict[str, float]:
        """CPU fallback for diversity calculation."""
        metrics = {}
        n_individuals = embeddings.shape[0]

        # Embedding diversity
        distances = []
        for i in range(n_individuals):
            for j in range(i + 1, n_individuals):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)

        if distances:
            metrics['embedding_diversity'] = float(np.mean(distances))
            metrics['embedding_diversity_std'] = float(np.std(distances))
            metrics['min_pairwise_distance'] = float(np.min(distances))
            metrics['max_pairwise_distance'] = float(np.max(distances))

        # Fitness diversity
        if fitness_scores is not None:
            metrics['fitness_mean'] = float(np.mean(fitness_scores))
            metrics['fitness_std'] = float(np.std(fitness_scores))
            metrics['fitness_variance'] = float(np.var(fitness_scores))
            metrics['fitness_range'] = float(np.max(fitness_scores) - np.min(fitness_scores))

        # Add uniqueness ratio (simplified for CPU)
        metrics['uniqueness_ratio'] = 1.0  # Assume all unique in CPU mode

        return metrics

    def _calculate_convergence_cpu(
        self,
        population_history: list[np.ndarray],
        fitness_history: list[np.ndarray]
    ) -> dict[str, float]:
        """CPU fallback for convergence metrics."""
        metrics = {}

        if len(fitness_history) > 1:
            best_fitness = [np.max(f) for f in fitness_history]
            avg_fitness = [np.mean(f) for f in fitness_history]

            best_improvements = np.diff(best_fitness)
            avg_improvements = np.diff(avg_fitness)

            metrics['best_fitness_improvement_rate'] = float(np.mean(best_improvements))
            metrics['avg_fitness_improvement_rate'] = float(np.mean(avg_improvements))
            metrics['fitness_stagnation_ratio'] = float(np.mean(best_improvements <= 1e-6))

        return metrics
