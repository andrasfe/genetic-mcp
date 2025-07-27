"""GPU-optimized selection strategies and diversity calculations for genetic algorithms.

This module provides GPU-accelerated implementations of advanced selection strategies
including Boltzmann selection, Stochastic Universal Sampling (SUS), rank-based selection,
and diversity metrics like crowding distance and fitness sharing.
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
    logging.warning("PyTorch not available, GPU selection optimization disabled")

CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUOptimizedSelection:
    """GPU-optimized selection strategies for genetic algorithms."""

    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.memory_manager = GPUMemoryManager(self.config)
        self.use_gpu = self.memory_manager.is_gpu_available()
        self.device = torch.device(self.config.device) if TORCH_AVAILABLE else None

        # Selection parameters
        self.temperature = 1.0  # For Boltzmann selection
        self.selection_pressure = 2.0  # For rank-based selection

        logger.info(f"Initialized GPU-optimized selection on {self.config.device}")

    def boltzmann_selection_batch(
        self,
        fitness_scores: np.ndarray,
        temperature: float,
        num_selections: int,
        batch_size: int = 1024
    ) -> np.ndarray:
        """GPU-accelerated Boltzmann selection for multiple selections.

        Mathematical basis: P(i) = exp(f_i / T) / sum(exp(f_j / T))
        where f_i is fitness, T is temperature
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._boltzmann_selection_cpu(fitness_scores, temperature, num_selections)

        # Convert to GPU tensor
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)

        # Normalize fitness to prevent overflow
        fitness_normalized = fitness_tensor - fitness_tensor.mean()

        # Calculate Boltzmann probabilities
        exp_fitness = torch.exp(fitness_normalized / temperature)
        probabilities = exp_fitness / exp_fitness.sum()

        # Batch selection for efficiency
        selections = []
        for i in range(0, num_selections, batch_size):
            batch_selections = min(batch_size, num_selections - i)

            # Use multinomial sampling
            selected_indices = torch.multinomial(
                probabilities,
                batch_selections,
                replacement=True
            )
            selections.append(selected_indices)

        # Concatenate and return
        all_selections = torch.cat(selections)
        return all_selections.cpu().numpy()

    def stochastic_universal_sampling_gpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int
    ) -> np.ndarray:
        """GPU-accelerated Stochastic Universal Sampling (SUS).

        Mathematical basis: Evenly spaced pointers reduce selection bias
        compared to roulette wheel selection.
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._sus_selection_cpu(fitness_scores, num_selections)

        n = len(fitness_scores)
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)

        # Calculate cumulative fitness
        total_fitness = fitness_tensor.sum()
        if total_fitness == 0:
            # Random selection if all fitness is zero
            return torch.randint(0, n, (num_selections,)).numpy()

        # Normalize and calculate cumulative probabilities
        probabilities = fitness_tensor / total_fitness
        cumulative_probs = torch.cumsum(probabilities, dim=0)

        # Generate evenly spaced pointers
        spacing = 1.0 / num_selections
        start = torch.rand(1, device=self.device).item() * spacing
        pointers = torch.arange(num_selections, device=self.device) * spacing + start

        # Find indices using searchsorted
        selected_indices = torch.searchsorted(cumulative_probs, pointers)

        # Clamp to valid range
        selected_indices = torch.clamp(selected_indices, 0, n - 1)

        return selected_indices.cpu().numpy()

    def rank_based_selection_gpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int,
        selection_pressure: float | None = None
    ) -> np.ndarray:
        """GPU-accelerated rank-based selection.

        Mathematical basis: P(i) = (2 - SP + 2(SP - 1)(i - 1)/(N - 1)) / N
        where SP is selection pressure (1.0 to 2.0), i is rank, N is population size
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._rank_selection_cpu(fitness_scores, num_selections, selection_pressure)

        if selection_pressure is None:
            selection_pressure = self.selection_pressure

        n = len(fitness_scores)
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)

        # Get sorted indices (ascending order)
        sorted_indices = torch.argsort(fitness_tensor)

        # Calculate rank-based probabilities
        ranks = torch.arange(1, n + 1, device=self.device).float()

        # Linear ranking formula
        probabilities = (2 - selection_pressure +
                        2 * (selection_pressure - 1) * (ranks - 1) / (n - 1)) / n

        # Ensure probabilities sum to 1
        probabilities = probabilities / probabilities.sum()

        # Sample based on rank probabilities
        rank_selections = torch.multinomial(probabilities, num_selections, replacement=True)

        # Map back to original indices
        selected_indices = sorted_indices[rank_selections]

        return selected_indices.cpu().numpy()

    def fitness_sharing_gpu(
        self,
        fitness_scores: np.ndarray,
        embeddings: np.ndarray,
        sigma_share: float = 0.1,
        alpha: float = 1.0
    ) -> np.ndarray:
        """GPU-accelerated fitness sharing for diversity preservation.

        Mathematical basis: Shared fitness = raw fitness / niche count
        Niche count = sum of sharing function over all individuals
        Sharing function: sh(d) = 1 - (d/sigma)^alpha if d < sigma, else 0
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._fitness_sharing_cpu(fitness_scores, embeddings, sigma_share, alpha)

        len(fitness_scores)
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)

        # Compute pairwise distances
        distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)

        # Apply sharing function
        # sh(d) = 1 - (d/sigma)^alpha if d < sigma, else 0
        sharing_values = torch.where(
            distances < sigma_share,
            1 - torch.pow(distances / sigma_share, alpha),
            torch.zeros_like(distances)
        )

        # Calculate niche counts (sum of sharing values for each individual)
        niche_counts = sharing_values.sum(dim=1)

        # Prevent division by zero
        niche_counts = torch.clamp(niche_counts, min=1e-6)

        # Calculate shared fitness
        shared_fitness = fitness_tensor / niche_counts

        return shared_fitness.cpu().numpy()

    def crowding_distance_gpu(
        self,
        objective_values: np.ndarray,
        pareto_ranks: np.ndarray | None = None
    ) -> np.ndarray:
        """GPU-accelerated crowding distance calculation for NSGA-II.

        Mathematical basis: Crowding distance = sum of normalized distances
        to neighbors in each objective dimension.
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._crowding_distance_cpu(objective_values, pareto_ranks)

        n, m = objective_values.shape  # n individuals, m objectives
        obj_tensor = torch.from_numpy(objective_values).float().to(self.device)

        # Initialize distances
        crowding_distances = torch.zeros(n, device=self.device)

        if pareto_ranks is not None:
            ranks_tensor = torch.from_numpy(pareto_ranks).to(self.device)
            unique_ranks = torch.unique(ranks_tensor)

            # Calculate crowding distance for each Pareto front
            for rank in unique_ranks:
                mask = ranks_tensor == rank
                front_indices = torch.where(mask)[0]

                if len(front_indices) <= 2:
                    # Boundary points get infinite distance
                    crowding_distances[front_indices] = float('inf')
                else:
                    # Calculate for this front
                    front_distances = self._calculate_front_crowding_gpu(
                        obj_tensor[front_indices],
                        front_indices
                    )
                    crowding_distances[front_indices] = front_distances
        else:
            # Calculate for entire population
            crowding_distances = self._calculate_front_crowding_gpu(obj_tensor, torch.arange(n))

        return crowding_distances.cpu().numpy()

    def _calculate_front_crowding_gpu(
        self,
        objectives: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Calculate crowding distance for a single front on GPU."""
        n, m = objectives.shape
        distances = torch.zeros(n, device=self.device)

        # For each objective
        for obj_idx in range(m):
            # Sort by this objective
            sorted_indices = torch.argsort(objectives[:, obj_idx])

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Calculate range for normalization
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]

            if obj_range > 0:
                # Calculate distances for middle points
                for i in range(1, n - 1):
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]

                    # Normalized distance to neighbors
                    distance = (objectives[next_idx, obj_idx] - objectives[prev_idx, obj_idx]) / obj_range
                    distances[curr_idx] += distance

        return distances

    def diversity_preservation_selection_gpu(
        self,
        fitness_scores: np.ndarray,
        embeddings: np.ndarray,
        num_selections: int,
        diversity_weight: float = 0.5
    ) -> np.ndarray:
        """GPU-accelerated selection balancing fitness and diversity.

        Combines fitness-based selection with diversity preservation
        using a weighted combination of fitness and distance metrics.
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._diversity_selection_cpu(
                fitness_scores, embeddings, num_selections, diversity_weight
            )

        n = len(fitness_scores)
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)

        # Normalize fitness scores
        fitness_normalized = (fitness_tensor - fitness_tensor.min()) / (fitness_tensor.max() - fitness_tensor.min() + 1e-6)

        selected_indices = []
        available_mask = torch.ones(n, dtype=torch.bool, device=self.device)

        # Greedy selection for diversity
        for _ in range(num_selections):
            if not available_mask.any():
                break

            available_indices = torch.where(available_mask)[0]

            if len(selected_indices) == 0:
                # First selection: choose best fitness
                best_idx = available_indices[torch.argmax(fitness_normalized[available_indices])]
            else:
                # Calculate minimum distance to already selected
                selected_embeddings = embeddings_tensor[torch.tensor(selected_indices, device=self.device)]
                distances = torch.cdist(
                    embeddings_tensor[available_indices].unsqueeze(0),
                    selected_embeddings.unsqueeze(0)
                ).squeeze(0)

                min_distances = distances.min(dim=1)[0]

                # Normalize distances
                if min_distances.max() > 0:
                    diversity_scores = min_distances / min_distances.max()
                else:
                    diversity_scores = torch.zeros_like(min_distances)

                # Combine fitness and diversity
                combined_scores = ((1 - diversity_weight) * fitness_normalized[available_indices] +
                                 diversity_weight * diversity_scores)

                # Select best combined score
                best_local_idx = torch.argmax(combined_scores)
                best_idx = available_indices[best_local_idx]

            selected_indices.append(best_idx.item())
            available_mask[best_idx] = False

        return np.array(selected_indices)

    def batch_tournament_selection_advanced_gpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int,
        tournament_size: int = 3,
        selection_pressure: float = 0.9
    ) -> np.ndarray:
        """GPU-accelerated tournament selection with selection pressure.

        Selection pressure controls probability of selecting best individual
        in tournament (1.0 = always select best, 0.5 = random).
        """
        if not self.use_gpu or not TORCH_AVAILABLE:
            return self._tournament_selection_cpu(
                fitness_scores, num_selections, tournament_size, selection_pressure
            )

        n = len(fitness_scores)
        fitness_tensor = torch.from_numpy(fitness_scores).float().to(self.device)

        # Generate all tournaments at once
        tournaments = torch.randint(0, n, (num_selections, tournament_size), device=self.device)

        # Get fitness values for all tournaments
        tournament_fitness = fitness_tensor[tournaments]

        # Sort tournaments by fitness
        sorted_fitness, sorted_indices = torch.sort(tournament_fitness, dim=1, descending=True)

        # Apply selection pressure
        if selection_pressure < 1.0:
            # Probabilistic selection based on rank
            rank_probs = torch.zeros(tournament_size, device=self.device)
            for i in range(tournament_size):
                rank_probs[i] = selection_pressure * (1 - selection_pressure) ** i

            # Normalize probabilities
            rank_probs = rank_probs / rank_probs.sum()

            # Sample ranks for each tournament
            selected_ranks = torch.multinomial(rank_probs, num_selections, replacement=True)

            # Get selected individuals
            batch_indices = torch.arange(num_selections, device=self.device)
            selected_local_indices = sorted_indices[batch_indices, selected_ranks]
            selected_indices = tournaments[batch_indices, selected_local_indices]
        else:
            # Always select best (standard tournament)
            selected_indices = tournaments[torch.arange(num_selections), sorted_indices[:, 0]]

        return selected_indices.cpu().numpy()

    # CPU fallback methods
    def _boltzmann_selection_cpu(
        self,
        fitness_scores: np.ndarray,
        temperature: float,
        num_selections: int
    ) -> np.ndarray:
        """CPU fallback for Boltzmann selection."""
        # Normalize to prevent overflow
        fitness_normalized = fitness_scores - np.mean(fitness_scores)

        # Calculate probabilities
        exp_fitness = np.exp(fitness_normalized / temperature)
        probabilities = exp_fitness / np.sum(exp_fitness)

        # Select indices
        return np.random.choice(len(fitness_scores), size=num_selections, p=probabilities)

    def _sus_selection_cpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int
    ) -> np.ndarray:
        """CPU fallback for Stochastic Universal Sampling."""
        n = len(fitness_scores)
        total_fitness = np.sum(fitness_scores)

        if total_fitness == 0:
            return np.random.choice(n, size=num_selections, replace=False)

        # Calculate cumulative probabilities
        probabilities = fitness_scores / total_fitness
        cumulative_probs = np.cumsum(probabilities)

        # Generate evenly spaced pointers
        spacing = 1.0 / num_selections
        start = np.random.random() * spacing
        pointers = np.arange(num_selections) * spacing + start

        # Select individuals
        selected_indices = []
        for pointer in pointers:
            idx = np.searchsorted(cumulative_probs, pointer)
            selected_indices.append(min(idx, n - 1))

        return np.array(selected_indices)

    def _rank_selection_cpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int,
        selection_pressure: float | None = None
    ) -> np.ndarray:
        """CPU fallback for rank-based selection."""
        if selection_pressure is None:
            selection_pressure = self.selection_pressure

        n = len(fitness_scores)
        sorted_indices = np.argsort(fitness_scores)

        # Calculate rank probabilities
        ranks = np.arange(1, n + 1)
        probabilities = (2 - selection_pressure +
                        2 * (selection_pressure - 1) * (ranks - 1) / (n - 1)) / n

        # Normalize
        probabilities = probabilities / probabilities.sum()

        # Sample
        rank_selections = np.random.choice(n, size=num_selections, p=probabilities)
        return sorted_indices[rank_selections]

    def _fitness_sharing_cpu(
        self,
        fitness_scores: np.ndarray,
        embeddings: np.ndarray,
        sigma_share: float,
        alpha: float
    ) -> np.ndarray:
        """CPU fallback for fitness sharing."""
        n = len(fitness_scores)
        shared_fitness = fitness_scores.copy()

        for i in range(n):
            niche_count = 0
            for j in range(n):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                if distance < sigma_share:
                    niche_count += 1 - (distance / sigma_share) ** alpha

            if niche_count > 0:
                shared_fitness[i] = fitness_scores[i] / niche_count

        return shared_fitness

    def _crowding_distance_cpu(
        self,
        objective_values: np.ndarray,
        pareto_ranks: np.ndarray | None = None
    ) -> np.ndarray:
        """CPU fallback for crowding distance calculation."""
        n, m = objective_values.shape
        crowding_distances = np.zeros(n)

        if pareto_ranks is not None:
            # Calculate for each front separately
            for rank in np.unique(pareto_ranks):
                front_indices = np.where(pareto_ranks == rank)[0]
                if len(front_indices) <= 2:
                    crowding_distances[front_indices] = float('inf')
                else:
                    front_distances = self._calculate_front_crowding_cpu(
                        objective_values[front_indices],
                        front_indices
                    )
                    crowding_distances[front_indices] = front_distances
        else:
            crowding_distances = self._calculate_front_crowding_cpu(
                objective_values, np.arange(n)
            )

        return crowding_distances

    def _calculate_front_crowding_cpu(
        self,
        objectives: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Calculate crowding distance for a single front on CPU."""
        n, m = objectives.shape
        distances = np.zeros(n)

        for obj_idx in range(m):
            sorted_indices = np.argsort(objectives[:, obj_idx])

            # Boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Range for normalization
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]

            if obj_range > 0:
                for i in range(1, n - 1):
                    curr_idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]

                    distance = (objectives[next_idx, obj_idx] - objectives[prev_idx, obj_idx]) / obj_range
                    distances[curr_idx] += distance

        return distances

    def _diversity_selection_cpu(
        self,
        fitness_scores: np.ndarray,
        embeddings: np.ndarray,
        num_selections: int,
        diversity_weight: float
    ) -> np.ndarray:
        """CPU fallback for diversity preservation selection."""
        n = len(fitness_scores)

        # Normalize fitness
        fitness_normalized = (fitness_scores - fitness_scores.min()) / (fitness_scores.max() - fitness_scores.min() + 1e-6)

        selected_indices = []
        available_indices = list(range(n))

        for _ in range(num_selections):
            if not available_indices:
                break

            if len(selected_indices) == 0:
                # First selection: best fitness
                best_idx = available_indices[np.argmax(fitness_normalized[available_indices])]
            else:
                # Calculate distances to selected
                min_distances = []
                for idx in available_indices:
                    distances = [np.linalg.norm(embeddings[idx] - embeddings[sel_idx])
                               for sel_idx in selected_indices]
                    min_distances.append(min(distances))

                min_distances = np.array(min_distances)

                # Normalize distances
                if min_distances.max() > 0:
                    diversity_scores = min_distances / min_distances.max()
                else:
                    diversity_scores = np.zeros_like(min_distances)

                # Combine scores
                available_fitness = fitness_normalized[available_indices]
                combined_scores = ((1 - diversity_weight) * available_fitness +
                                 diversity_weight * diversity_scores)

                # Select best
                best_local_idx = np.argmax(combined_scores)
                best_idx = available_indices[best_local_idx]

            selected_indices.append(best_idx)
            available_indices.remove(best_idx)

        return np.array(selected_indices)

    def _tournament_selection_cpu(
        self,
        fitness_scores: np.ndarray,
        num_selections: int,
        tournament_size: int,
        selection_pressure: float
    ) -> np.ndarray:
        """CPU fallback for tournament selection with pressure."""
        n = len(fitness_scores)
        selected = []

        for _ in range(num_selections):
            # Create tournament
            tournament_indices = np.random.choice(n, size=tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]

            # Sort by fitness
            sorted_indices = np.argsort(tournament_fitness)[::-1]

            # Apply selection pressure
            if selection_pressure < 1.0:
                # Probabilistic selection
                probs = [selection_pressure * (1 - selection_pressure) ** i
                        for i in range(tournament_size)]
                probs = np.array(probs) / sum(probs)

                selected_rank = np.random.choice(tournament_size, p=probs)
                selected_idx = tournament_indices[sorted_indices[selected_rank]]
            else:
                # Always select best
                selected_idx = tournament_indices[sorted_indices[0]]

            selected.append(selected_idx)

        return np.array(selected)
