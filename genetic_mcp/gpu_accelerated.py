"""GPU-accelerated operations for genetic algorithm MCP server.

This module provides GPU-accelerated implementations for computationally intensive
operations including embedding generation, fitness evaluation, and genetic operations.
Falls back to CPU when GPU is not available.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

# GPU libraries with fallback imports
try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU acceleration disabled")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available, some GPU operations will use PyTorch only")

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Configuration for GPU operations."""
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_sequence_length: int = 512
    embedding_dim: int = 768  # Default for sentence-transformers
    use_mixed_precision: bool = True
    memory_fraction: float = 0.8  # Fraction of GPU memory to use


class GPUMemoryManager:
    """Manages GPU memory allocation and monitoring."""

    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = torch.device(config.device) if TORCH_AVAILABLE else None

        if self.is_gpu_available():
            # Set memory fraction
            if TORCH_AVAILABLE:
                torch.cuda.set_per_process_memory_fraction(config.memory_fraction)

            # Initialize memory pool for CuPy
            if CUPY_AVAILABLE and config.device == "cuda":
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=int(torch.cuda.get_device_properties(0).total_memory * config.memory_fraction))

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    def get_memory_stats(self) -> dict[str, float]:
        """Get current GPU memory statistics."""
        if not self.is_gpu_available():
            return {"allocated": 0, "reserved": 0, "free": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        }

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
        if CUPY_AVAILABLE and self.config.device == "cuda":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()


class GPUEmbeddingGenerator:
    """GPU-accelerated embedding generation using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self.memory_manager = GPUMemoryManager(self.config)

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for embedding generation")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.config.device)
        self.model.max_seq_length = self.config.max_sequence_length

        # Enable mixed precision if available
        if self.config.use_mixed_precision and self.config.device == "cuda":
            self.model.half()

        # Cache for embeddings
        self.embedding_cache: dict[str, torch.Tensor] = {}
        self._cache_lock = asyncio.Lock()

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"Initialized embedding generator on {self.config.device}")

    async def generate_embeddings(self, texts: list[str], ids: list[str]) -> dict[str, np.ndarray]:
        """Generate embeddings for a batch of texts with caching."""
        async with self._cache_lock:
            # Check cache
            results = {}
            uncached_texts = []
            uncached_ids = []

            for text, id_ in zip(texts, ids, strict=False):
                if id_ in self.embedding_cache:
                    results[id_] = self.embedding_cache[id_].cpu().numpy()
                else:
                    uncached_texts.append(text)
                    uncached_ids.append(id_)

            if not uncached_texts:
                return results

        # Generate embeddings for uncached texts
        embeddings = await self._generate_batch(uncached_texts)

        # Update cache and results
        async with self._cache_lock:
            for id_, embedding in zip(uncached_ids, embeddings, strict=False):
                tensor_embedding = torch.from_numpy(embedding).to(self.config.device)
                self.embedding_cache[id_] = tensor_embedding
                results[id_] = embedding

        return results

    async def _generate_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._generate_embeddings_sync,
            texts
        )
        return embeddings

    def _generate_embeddings_sync(self, texts: list[str]) -> np.ndarray:
        """Synchronous embedding generation."""
        with torch.no_grad():
            # Process in batches
            all_embeddings = []

            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    batch_size=len(batch),
                    show_progress_bar=False
                )
                all_embeddings.append(embeddings)

            return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.memory_manager.clear_cache()


class GPUFitnessEvaluator:
    """GPU-accelerated fitness evaluation."""

    def __init__(self, config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self.memory_manager = GPUMemoryManager(self.config)
        self.use_gpu = self.memory_manager.is_gpu_available()

        # Choose backend
        if self.use_gpu and CUPY_AVAILABLE:
            self.xp = cp
            logger.info("Using CuPy for fitness evaluation")
        elif self.use_gpu and TORCH_AVAILABLE:
            self.xp = None  # Use PyTorch
            logger.info("Using PyTorch for fitness evaluation")
        else:
            self.xp = np
            logger.info("Using NumPy for fitness evaluation (CPU)")

    def batch_cosine_similarity(self, embeddings: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings and target in batch."""
        if self.use_gpu and TORCH_AVAILABLE:
            # Convert to PyTorch tensors
            emb_tensor = torch.from_numpy(embeddings).to(self.config.device)
            target_tensor = torch.from_numpy(target).to(self.config.device)

            # Normalize
            emb_norm = F.normalize(emb_tensor, p=2, dim=1)
            target_norm = F.normalize(target_tensor.unsqueeze(0), p=2, dim=1)

            # Compute similarity
            similarities = torch.mm(emb_norm, target_norm.t()).squeeze()
            return similarities.cpu().numpy()

        elif self.use_gpu and CUPY_AVAILABLE:
            # Use CuPy
            emb_gpu = cp.asarray(embeddings)
            target_gpu = cp.asarray(target)

            # Normalize
            emb_norm = emb_gpu / cp.linalg.norm(emb_gpu, axis=1, keepdims=True)
            target_norm = target_gpu / cp.linalg.norm(target_gpu)

            # Compute similarity
            similarities = cp.dot(emb_norm, target_norm)
            return cp.asnumpy(similarities)

        else:
            # CPU fallback
            emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            target_norm = target / np.linalg.norm(target)
            return np.dot(emb_norm, target_norm)

    def compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between all embeddings."""
        if self.use_gpu and TORCH_AVAILABLE:
            emb_tensor = torch.from_numpy(embeddings).to(self.config.device)
            distances = torch.cdist(emb_tensor, emb_tensor, p=2)
            return distances.cpu().numpy()

        elif self.use_gpu and CUPY_AVAILABLE:
            emb_gpu = cp.asarray(embeddings)
            # Efficient pairwise distance computation
            distances = cp.sqrt(
                cp.sum(emb_gpu**2, axis=1, keepdims=True) +
                cp.sum(emb_gpu**2, axis=1) -
                2 * cp.dot(emb_gpu, emb_gpu.T)
            )
            return cp.asnumpy(distances)

        else:
            # CPU fallback using broadcasting
            diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
            return np.sqrt(np.sum(diff**2, axis=2))

    def compute_novelty_scores(self, embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """Compute novelty scores based on k-nearest neighbors."""
        distances = self.compute_pairwise_distances(embeddings)
        n = distances.shape[0]

        if self.use_gpu and TORCH_AVAILABLE:
            dist_tensor = torch.from_numpy(distances).to(self.config.device)
            # Set diagonal to infinity to exclude self
            dist_tensor.fill_diagonal_(float('inf'))

            # Find k-nearest neighbors
            k = min(k, n - 1)
            knn_distances, _ = torch.topk(dist_tensor, k, largest=False, dim=1)
            novelty_scores = knn_distances.mean(dim=1)

            # Normalize
            max_score = novelty_scores.max()
            if max_score > 0:
                novelty_scores = novelty_scores / max_score

            return novelty_scores.cpu().numpy()

        elif self.use_gpu and CUPY_AVAILABLE:
            dist_gpu = cp.asarray(distances)
            cp.fill_diagonal(dist_gpu, cp.inf)

            k = min(k, n - 1)
            # Sort and take k smallest
            sorted_indices = cp.argsort(dist_gpu, axis=1)
            knn_indices = sorted_indices[:, :k]

            # Gather k-nearest distances
            knn_distances = cp.take_along_axis(dist_gpu, knn_indices, axis=1)
            novelty_scores = knn_distances.mean(axis=1)

            # Normalize
            max_score = novelty_scores.max()
            if max_score > 0:
                novelty_scores = novelty_scores / max_score

            return cp.asnumpy(novelty_scores)

        else:
            # CPU fallback
            np.fill_diagonal(distances, np.inf)
            k = min(k, n - 1)

            # Find k-nearest neighbors for each point
            knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
            knn_distances = np.take_along_axis(distances, knn_indices, axis=1)
            novelty_scores = knn_distances.mean(axis=1)

            # Normalize
            max_score = novelty_scores.max()
            if max_score > 0:
                novelty_scores = novelty_scores / max_score

            return novelty_scores

    def batch_evaluate_fitness(
        self,
        embeddings: np.ndarray,
        target_embedding: np.ndarray,
        weights: dict[str, float],
        feasibility_scores: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Evaluate fitness for entire population in batch."""
        n = embeddings.shape[0]

        # Compute relevance scores
        relevance_scores = (self.batch_cosine_similarity(embeddings, target_embedding) + 1) / 2

        # Compute novelty scores
        novelty_scores = self.compute_novelty_scores(embeddings)

        # Use provided feasibility scores or default
        if feasibility_scores is None:
            feasibility_scores = np.full(n, 0.7)  # Default feasibility

        # Compute weighted fitness
        fitness_scores = (
            weights['relevance'] * relevance_scores +
            weights['novelty'] * novelty_scores +
            weights['feasibility'] * feasibility_scores
        )

        components = {
            'relevance': relevance_scores,
            'novelty': novelty_scores,
            'feasibility': feasibility_scores
        }

        return fitness_scores, components


class GPUGeneticOperations:
    """GPU-accelerated genetic algorithm operations."""

    def __init__(self, config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self.memory_manager = GPUMemoryManager(self.config)
        self.use_gpu = self.memory_manager.is_gpu_available()

    def tournament_selection_batch(
        self,
        fitness_scores: np.ndarray,
        num_selections: int,
        tournament_size: int = 3
    ) -> np.ndarray:
        """Perform batch tournament selection on GPU."""
        n = len(fitness_scores)

        if self.use_gpu and TORCH_AVAILABLE:
            fitness_tensor = torch.from_numpy(fitness_scores).to(self.config.device)

            # Generate random tournaments
            tournaments = torch.randint(0, n, (num_selections, tournament_size), device=self.config.device)

            # Get fitness scores for tournaments
            tournament_fitness = fitness_tensor[tournaments]

            # Select winners (highest fitness in each tournament)
            _, winners_idx = tournament_fitness.max(dim=1)
            winners = tournaments[torch.arange(num_selections), winners_idx]

            return winners.cpu().numpy()

        elif self.use_gpu and CUPY_AVAILABLE:
            fitness_gpu = cp.asarray(fitness_scores)

            # Generate random tournaments
            tournaments = cp.random.randint(0, n, (num_selections, tournament_size))

            # Get fitness scores for tournaments
            tournament_fitness = fitness_gpu[tournaments]

            # Select winners
            winners_idx = tournament_fitness.argmax(axis=1)
            winners = tournaments[cp.arange(num_selections), winners_idx]

            return cp.asnumpy(winners)

        else:
            # CPU fallback
            winners = []
            for _ in range(num_selections):
                tournament = np.random.choice(n, size=tournament_size, replace=False)
                winner = tournament[np.argmax(fitness_scores[tournament])]
                winners.append(winner)
            return np.array(winners)

    def parallel_crossover_indices(
        self,
        parent_pairs: list[tuple[int, int]],
        sequence_lengths: np.ndarray,
        crossover_rate: float
    ) -> list[tuple[bool, int, int]]:
        """Generate crossover points for multiple parent pairs in parallel."""
        num_pairs = len(parent_pairs)

        if self.use_gpu and TORCH_AVAILABLE:
            # Generate crossover decisions
            do_crossover = torch.rand(num_pairs, device=self.config.device) < crossover_rate

            # Generate crossover points
            points1 = []
            points2 = []

            for (p1_idx, p2_idx) in parent_pairs:
                len1 = sequence_lengths[p1_idx]
                len2 = sequence_lengths[p2_idx]

                point1 = torch.randint(1, max(2, len1), (1,), device=self.config.device).item()
                point2 = torch.randint(1, max(2, len2), (1,), device=self.config.device).item()

                points1.append(point1)
                points2.append(point2)

            return list(zip(do_crossover.cpu().numpy(), points1, points2, strict=False))

        else:
            # CPU fallback
            results = []
            for _i, (p1_idx, p2_idx) in enumerate(parent_pairs):
                do_crossover = np.random.random() < crossover_rate

                len1 = sequence_lengths[p1_idx]
                len2 = sequence_lengths[p2_idx]

                point1 = np.random.randint(1, max(2, len1))
                point2 = np.random.randint(1, max(2, len2))

                results.append((do_crossover, point1, point2))

            return results

    def batch_mutation_mask(
        self,
        population_size: int,
        mutation_rate: float,
        sequence_lengths: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate mutation masks for entire population."""
        if self.use_gpu and TORCH_AVAILABLE:
            # Generate mutation decisions
            mutation_mask = torch.rand(population_size, device=self.config.device) < mutation_rate
            return mutation_mask.cpu().numpy()

        elif self.use_gpu and CUPY_AVAILABLE:
            mutation_mask = cp.random.random(population_size) < mutation_rate
            return cp.asnumpy(mutation_mask)

        else:
            return np.random.random(population_size) < mutation_rate


class GPUBatchProcessor:
    """Manages batch processing with GPU acceleration."""

    def __init__(self, config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self.embedding_generator = GPUEmbeddingGenerator(config=self.config)
        self.fitness_evaluator = GPUFitnessEvaluator(config=self.config)
        self.genetic_ops = GPUGeneticOperations(config=self.config)

        logger.info(f"Initialized GPU batch processor on {self.config.device}")

    async def process_population_batch(
        self,
        ideas: list[dict[str, Any]],
        target_prompt: str,
        fitness_weights: dict[str, float]
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Process entire population in batch for maximum GPU efficiency."""
        # Extract texts and IDs
        texts = [idea['content'] for idea in ideas]
        ids = [idea['id'] for idea in ideas]

        # Add target prompt
        all_texts = texts + [target_prompt]
        all_ids = ids + ['target']

        # Generate embeddings for all texts
        embeddings_dict = await self.embedding_generator.generate_embeddings(all_texts, all_ids)

        # Separate idea embeddings and target embedding
        idea_embeddings = np.vstack([embeddings_dict[id_] for id_ in ids])
        target_embedding = embeddings_dict['target']

        # Evaluate fitness in batch
        fitness_scores, score_components = self.fitness_evaluator.batch_evaluate_fitness(
            idea_embeddings,
            target_embedding,
            fitness_weights
        )

        return fitness_scores, score_components, embeddings_dict

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current GPU memory statistics."""
        return {
            "embedding_cache_size": len(self.embedding_generator.embedding_cache),
            "gpu_memory": self.embedding_generator.memory_manager.get_memory_stats()
        }

    def clear_caches(self) -> None:
        """Clear all GPU caches."""
        self.embedding_generator.clear_cache()
        self.fitness_evaluator.memory_manager.clear_cache()
        self.genetic_ops.memory_manager.clear_cache()


# Factory function for creating GPU-accelerated components
def create_gpu_accelerated_components(config: GPUConfig | None = None) -> dict[str, Any]:
    """Create all GPU-accelerated components with proper configuration."""
    config = config or GPUConfig()

    # Check GPU availability
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU acceleration. Install with: pip install torch sentence-transformers")

    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        config.device = "cpu"

    # Log GPU info if available
    if config.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.1f} GB memory")

    return {
        "config": config,
        "batch_processor": GPUBatchProcessor(config),
        "embedding_generator": GPUEmbeddingGenerator(config=config),
        "fitness_evaluator": GPUFitnessEvaluator(config=config),
        "genetic_ops": GPUGeneticOperations(config=config)
    }
