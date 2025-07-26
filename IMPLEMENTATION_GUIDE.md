# Implementation Guide for Genetic Algorithm MCP Server

## Quick Start Implementation Plan

Based on our architectural design and expert insights, here's a practical implementation guide that addresses all key requirements.

## Core Implementation Strategy

### 1. Project Structure
```
genetic_mcp/
├── src/
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server core
│   │   ├── protocol.py         # Protocol message definitions
│   │   └── transport/
│   │       ├── stdio.py        # Stdio transport
│   │       └── http_sse.py    # HTTP/SSE transport
│   ├── genetic/
│   │   ├── __init__.py
│   │   ├── engine.py           # Genetic algorithm engine
│   │   ├── operators.py        # Selection, crossover, mutation
│   │   └── population.py       # Population management
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── pool.py             # Worker pool manager
│   │   ├── llm_worker.py       # LLM worker implementation
│   │   └── rate_limiter.py     # Rate limiting logic
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── fitness.py          # Multi-objective fitness
│   │   ├── metrics.py          # Metric calculations
│   │   └── normalization.py    # Score normalization
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py          # Session management
│   │   └── state.py            # State persistence
│   └── utils/
│       ├── __init__.py
│       ├── encoding.py         # Idea encoding/decoding
│       ├── lineage.py          # Lineage tracking
│       └── monitoring.py       # Metrics collection
├── tests/
├── examples/
└── requirements.txt
```

### 2. Key Implementation Details

#### Parallel LLM Workers (Async Pattern)
```python
# Based on expert recommendation for I/O-bound LLM calls
import asyncio
from typing import List, Optional
import aiohttp
from asyncio import Semaphore

class LLMWorkerPool:
    def __init__(self, max_workers: int = 50, rate_limit: int = 100):
        self.semaphore = Semaphore(max_workers)
        self.rate_limiter = TokenBucket(rate_limit)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def generate_idea(self, prompt: str, temperature: float = 0.8) -> str:
        async with self.semaphore:  # Control concurrency
            await self.rate_limiter.acquire()  # Rate limiting
            
            try:
                async with self.session.post(
                    self.llm_endpoint,
                    json={"prompt": prompt, "temperature": temperature},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(self._backoff_time())
                        return await self.generate_idea(prompt, temperature)
                    
                    result = await response.json()
                    return result["text"]
                    
            except Exception as e:
                # Circuit breaker pattern
                self.error_count += 1
                if self.error_count > self.error_threshold:
                    await self._circuit_break()
                raise
                
    async def generate_batch(self, prompts: List[str]) -> List[str]:
        tasks = [self.generate_idea(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

#### Genetic Algorithm Implementation
```python
# Based on expert GA recommendations
from typing import List, Tuple
import numpy as np

class GeneticEngine:
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
    async def evolve_generation(self, population: Population) -> Population:
        # Evaluate fitness
        fitness_scores = await self.evaluate_population(population)
        
        # Elite selection (preserve top performers)
        elite_count = int(self.population_size * self.elitism_rate)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population = [population[i] for i in elite_indices]
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Tournament selection for diversity
            parent1 = self.tournament_select(population, fitness_scores)
            parent2 = self.tournament_select(population, fitness_scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = await self.semantic_crossover(parent1, parent2)
            else:
                child = parent1 if np.random.random() < 0.5 else parent2
                
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = await self.semantic_mutation(child)
                
            new_population.append(child)
            
        return Population(new_population)
        
    async def semantic_crossover(self, parent1: Idea, parent2: Idea) -> Idea:
        # Chunk-based crossover at semantic boundaries
        chunks1 = self.extract_semantic_chunks(parent1)
        chunks2 = self.extract_semantic_chunks(parent2)
        
        # Interleave chunks maintaining coherence
        new_chunks = []
        for i in range(max(len(chunks1), len(chunks2))):
            source = chunks1 if i % 2 == 0 else chunks2
            if i < len(source):
                new_chunks.append(source[i])
                
        return self.reconstruct_idea(new_chunks)
```

#### Multi-Objective Fitness Evaluation
```python
# Based on expert fitness function recommendations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import RobustScaler

@dataclass
class FitnessScore:
    relevance: float
    novelty: float
    feasibility: float
    combined: float
    
class AdaptiveFitnessEvaluator:
    def __init__(self):
        self.weights = {"relevance": 0.4, "novelty": 0.3, "feasibility": 0.3}
        self.scaler = RobustScaler()  # Handles outliers better
        self.history = []
        
    async def evaluate(self, idea: Idea, context: EvaluationContext) -> FitnessScore:
        # Compute raw metrics
        relevance = await self.compute_relevance(idea, context)
        novelty = await self.compute_novelty(idea, context)
        feasibility = await self.compute_feasibility(idea, context)
        
        # Normalize using robust scaling
        normalized = self.normalize_scores({
            "relevance": relevance,
            "novelty": novelty,
            "feasibility": feasibility
        })
        
        # Apply adaptive weights
        combined = sum(
            normalized[metric] * weight 
            for metric, weight in self.weights.items()
        )
        
        # Add diversity bonus to prevent gaming
        diversity_bonus = self.compute_diversity_bonus(idea, context)
        combined += diversity_bonus
        
        score = FitnessScore(
            relevance=normalized["relevance"],
            novelty=normalized["novelty"],
            feasibility=normalized["feasibility"],
            combined=combined
        )
        
        # Update adaptive weights based on performance
        self.update_weights(score, context)
        
        return score
        
    async def compute_novelty(self, idea: Idea, context: EvaluationContext) -> float:
        # Use sentence transformers for semantic space
        embedding = await self.encoder.encode(idea.text)
        
        # Find k-nearest neighbors
        neighbors = context.find_nearest_neighbors(embedding, k=5)
        
        # Novelty is average distance to neighbors
        if neighbors:
            distances = [self.cosine_distance(embedding, n) for n in neighbors]
            return np.mean(distances)
        return 1.0  # Maximum novelty if no neighbors
```

#### MCP Protocol Implementation
```python
# Based on expert MCP recommendations
from enum import Enum
from typing import Dict, Any, Optional
import json

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    
class MCPMessage:
    def __init__(self, 
                 msg_type: MessageType,
                 operation: str,
                 payload: Dict[str, Any],
                 session_id: Optional[str] = None,
                 msg_id: Optional[str] = None):
        self.version = "1.0"
        self.type = msg_type
        self.operation = operation
        self.payload = payload
        self.session_id = session_id
        self.id = msg_id or self._generate_id()
        self.metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_version": "1.0"
        }
        
class MCPServer:
    def __init__(self):
        self.handlers = {}
        self.session_manager = SessionManager()
        self.transport_adapters = {}
        
    def register_handler(self, operation: str, handler):
        self.handlers[operation] = handler
        
    async def handle_message(self, raw_message: str) -> str:
        try:
            # Parse and validate
            message = self.parse_message(raw_message)
            
            # Route to handler
            handler = self.handlers.get(message.operation)
            if not handler:
                return self.error_response(f"Unknown operation: {message.operation}")
                
            # Execute with session context
            session = None
            if message.session_id:
                session = await self.session_manager.get_session(message.session_id)
                
            result = await handler(message, session)
            
            # Format response
            return self.format_response(message, result)
            
        except Exception as e:
            return self.error_response(str(e), traceback=True)
```

#### Progress Streaming Implementation
```python
# For long-running operations
class ProgressStreamer:
    def __init__(self, transport: Transport):
        self.transport = transport
        self.active_streams = {}
        
    async def stream_progress(self, session_id: str, operation_id: str):
        stream_key = f"{session_id}:{operation_id}"
        self.active_streams[stream_key] = True
        
        try:
            generation = 0
            while self.active_streams.get(stream_key, False):
                # Get current progress
                progress = await self.get_progress(session_id)
                
                # Format progress update
                update = {
                    "type": "progress",
                    "session_id": session_id,
                    "operation_id": operation_id,
                    "generation": progress.generation,
                    "best_fitness": progress.best_fitness,
                    "population_diversity": progress.diversity,
                    "estimated_remaining": progress.eta
                }
                
                # Send via appropriate transport
                if isinstance(self.transport, HTTPSSETransport):
                    await self.transport.send_sse(json.dumps(update))
                else:
                    await self.transport.send_notification(update)
                    
                # Check for completion
                if progress.is_complete:
                    break
                    
                await asyncio.sleep(1)  # Throttle updates
                
        finally:
            del self.active_streams[stream_key]
```

### 3. Configuration and Deployment

#### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MCP_TRANSPORT=stdio
ENV REDIS_URL=redis://localhost:6379
ENV MAX_WORKERS=50

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import src.health; src.health.check()"

# Run server
CMD ["python", "-m", "src.mcp.server"]
```

#### Redis Configuration for State Management
```python
import redis.asyncio as redis
import json
from typing import Optional

class RedisStateStore:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.url = url
        self.pool = None
        
    async def connect(self):
        self.pool = redis.ConnectionPool.from_url(self.url)
        
    async def save_session(self, session_id: str, state: dict, ttl: int = 3600):
        async with redis.Redis(connection_pool=self.pool) as r:
            await r.setex(
                f"session:{session_id}",
                ttl,
                json.dumps(state)
            )
            
    async def get_session(self, session_id: str) -> Optional[dict]:
        async with redis.Redis(connection_pool=self.pool) as r:
            data = await r.get(f"session:{session_id}")
            return json.loads(data) if data else None
```

### 4. Performance Optimization Tips

#### Embedding Cache
```python
from functools import lru_cache
import hashlib

class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
        
    async def get_embedding(self, text: str) -> np.ndarray:
        text_hash = self._hash_text(text)
        
        if text_hash in self.cache:
            return self.cache[text_hash]
            
        # Compute embedding
        embedding = await self.encoder.encode(text)
        
        # LRU eviction
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1]["timestamp"])
            del self.cache[oldest[0]]
            
        self.cache[text_hash] = {
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        return embedding
```

### 5. Monitoring and Observability

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
generation_counter = Counter('ga_generations_total', 'Total GA generations')
fitness_histogram = Histogram('ga_fitness_scores', 'Fitness score distribution')
worker_gauge = Gauge('llm_workers_active', 'Active LLM workers')
api_latency = Histogram('llm_api_latency_seconds', 'LLM API call latency')

class MetricsCollector:
    @staticmethod
    def record_generation(fitness_scores: List[float]):
        generation_counter.inc()
        for score in fitness_scores:
            fitness_histogram.observe(score)
            
    @staticmethod
    async def track_api_call(func):
        start = time.time()
        try:
            worker_gauge.inc()
            result = await func()
            return result
        finally:
            worker_gauge.dec()
            api_latency.observe(time.time() - start)
```

## Next Steps

1. **Start with Phase 1**: Implement basic MCP server with stdio transport
2. **Add Worker Pool**: Implement async LLM worker management
3. **Build GA Engine**: Create core genetic algorithm operations
4. **Integrate Fitness**: Add multi-objective evaluation
5. **Add Streaming**: Implement progress updates
6. **Scale Testing**: Load test with multiple concurrent sessions

This implementation guide provides the foundation for building a production-ready genetic algorithm MCP server following all the best practices identified by our domain experts.