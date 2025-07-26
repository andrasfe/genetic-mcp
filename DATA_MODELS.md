# Data Models and Interfaces Specification

## Core Data Models

### 1. Idea Representation

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

@dataclass
class Idea:
    """Core representation of a generated idea"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    parent_ids: List[str] = field(default_factory=list)
    
    # Genetic operations applied
    mutations: List[str] = field(default_factory=list)
    crossover_type: Optional[str] = None
    
    # Evaluation scores
    fitness_scores: Optional['FitnessScore'] = None
    raw_scores: Dict[str, float] = field(default_factory=dict)
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class IdeaChromosome:
    """Genetic representation of an idea"""
    segments: List[str]  # Semantic chunks
    features: Dict[str, float]  # Extracted features
    structure: Dict[str, Any]  # Grammatical structure
    
    def to_text(self) -> str:
        """Reconstruct text from chromosome"""
        return " ".join(self.segments)
```

### 2. Population Management

```python
@dataclass
class Population:
    """Container for a generation of ideas"""
    generation: int
    ideas: List[Idea]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    diversity_score: float = 0.0
    convergence_rate: float = 0.0
    
    # Evolution parameters used
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 0.5
    
    def get_elite(self, percentage: float = 0.1) -> List[Idea]:
        """Get top performing ideas"""
        sorted_ideas = sorted(
            self.ideas, 
            key=lambda x: x.fitness_scores.combined if x.fitness_scores else 0,
            reverse=True
        )
        elite_count = max(1, int(len(self.ideas) * percentage))
        return sorted_ideas[:elite_count]
```

### 3. Fitness and Evaluation

```python
@dataclass
class FitnessScore:
    """Multi-objective fitness scores"""
    relevance: float  # 0-1, similarity to target
    novelty: float    # 0-1, distance from neighbors
    feasibility: float  # 0-1, critic rating
    combined: float   # Weighted combination
    
    # Component scores
    semantic_coherence: float = 0.0
    grammatical_quality: float = 0.0
    specificity: float = 0.0
    
    # Metadata
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    evaluator_version: str = "1.0"
    confidence: float = 1.0

@dataclass
class EvaluationContext:
    """Context for fitness evaluation"""
    target_prompt: str
    existing_ideas: List[Idea]
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Cached computations
    _embeddings_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _similarity_matrix: Optional[np.ndarray] = None
    
    def find_nearest_neighbors(self, embedding: np.ndarray, k: int = 5) -> List[np.ndarray]:
        """Find k-nearest neighbors in embedding space"""
        # Implementation details...
        pass
```

### 4. Session Management

```python
from enum import Enum

class SessionState(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    GENERATING = "generating"
    EVOLVING = "evolving"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"

@dataclass
class SessionConfig:
    """Configuration for a GA session"""
    # GA parameters
    population_size: int = 50
    max_generations: int = 100
    convergence_threshold: float = 0.95
    
    # Genetic operators
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    
    # Selection strategy
    selection_method: str = "tournament"
    tournament_size: int = 3
    
    # Fitness weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.4,
        "novelty": 0.3,
        "feasibility": 0.3
    })
    
    # LLM parameters
    llm_temperature: float = 0.8
    llm_model: str = "gpt-4"
    max_tokens: int = 150
    
    # Adaptive parameters
    adaptive_mutation: bool = True
    dynamic_fitness_weights: bool = False
    
    # Resource limits
    max_workers: int = 20
    timeout_seconds: int = 3600
    memory_limit_mb: int = 1024

@dataclass
class Session:
    """Active GA session"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: SessionConfig = field(default_factory=SessionConfig)
    state: SessionState = SessionState.CREATED
    
    # Runtime data
    current_generation: int = 0
    populations: List[Population] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Results
    best_ideas: List[Idea] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    
    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Checkpointing
    last_checkpoint: Optional[datetime] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
```

### 5. Worker and Task Models

```python
@dataclass
class GenerationTask:
    """Task for LLM worker"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 150
    
    # Context
    session_id: str = ""
    generation: int = 0
    parent_ideas: List[Idea] = field(default_factory=list)
    
    # Constraints
    avoid_phrases: List[str] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0
    retry_count: int = 0

@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_latency_ms: float = 0.0
    current_load: float = 0.0
    last_active: datetime = field(default_factory=datetime.utcnow)
    
class WorkerState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"
```

### 6. Protocol Messages

```python
@dataclass
class MCPRequest:
    """Base MCP request"""
    version: str = "1.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""
    session_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPResponse:
    """Base MCP response"""
    version: str = "1.0"
    id: str = ""
    request_id: str = ""
    success: bool = True
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProgressUpdate:
    """Progress notification"""
    session_id: str
    operation_id: str
    progress_percentage: float
    current_generation: int
    best_fitness: float
    population_diversity: float
    estimated_remaining_seconds: int
    message: str = ""
    
@dataclass
class CreateSessionRequest(MCPRequest):
    """Request to create new session"""
    operation: str = "create_session"
    
    def __post_init__(self):
        # Validate config in payload
        if "config" not in self.payload:
            self.payload["config"] = {}

@dataclass
class GenerateIdeasRequest(MCPRequest):
    """Request to generate ideas"""
    operation: str = "generate"
    
    def __post_init__(self):
        required = ["prompt", "count"]
        for field in required:
            if field not in self.payload:
                raise ValueError(f"Missing required field: {field}")

@dataclass
class EvolveRequest(MCPRequest):
    """Request to evolve population"""
    operation: str = "evolve"
    
    def __post_init__(self):
        if "generations" not in self.payload:
            self.payload["generations"] = 1
```

### 7. Lineage and History

```python
@dataclass
class LineageNode:
    """Node in lineage tree"""
    idea_id: str
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    
    # Evolution details
    operation: str = ""  # "mutation", "crossover", "elite"
    operation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    fitness_improvement: float = 0.0
    contribution_score: float = 0.0

@dataclass
class EvolutionHistory:
    """Complete evolution history"""
    session_id: str
    total_generations: int
    total_ideas_generated: int
    
    # Lineage tree
    lineage_nodes: Dict[str, LineageNode] = field(default_factory=dict)
    
    # Statistics over time
    fitness_progression: List[float] = field(default_factory=list)
    diversity_progression: List[float] = field(default_factory=list)
    
    # Notable events
    breakthroughs: List[Dict[str, Any]] = field(default_factory=list)
    stagnation_periods: List[Dict[str, Any]] = field(default_factory=list)
```

### 8. Error and Exception Models

```python
@dataclass
class MCPError:
    """Structured error information"""
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    traceback: Optional[str] = None
    
    # Error categories
    ERROR_CODES = {
        "SESSION_NOT_FOUND": "The requested session does not exist",
        "INVALID_OPERATION": "The requested operation is invalid",
        "RATE_LIMIT_EXCEEDED": "API rate limit has been exceeded",
        "WORKER_POOL_EXHAUSTED": "No available workers",
        "EVOLUTION_TIMEOUT": "Evolution process timed out",
        "INVALID_PARAMETERS": "Invalid parameters provided",
        "INTERNAL_ERROR": "Internal server error occurred"
    }
```

## Interface Contracts

### Worker Pool Interface
```python
from abc import ABC, abstractmethod

class IWorkerPool(ABC):
    @abstractmethod
    async def submit_task(self, task: GenerationTask) -> Idea:
        """Submit a generation task"""
        pass
        
    @abstractmethod
    async def scale_workers(self, target_count: int) -> None:
        """Scale worker pool"""
        pass
        
    @abstractmethod
    async def get_stats(self) -> Dict[str, WorkerStats]:
        """Get worker statistics"""
        pass
```

### Fitness Evaluator Interface
```python
class IFitnessEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, idea: Idea, context: EvaluationContext) -> FitnessScore:
        """Evaluate idea fitness"""
        pass
        
    @abstractmethod
    async def batch_evaluate(self, ideas: List[Idea], context: EvaluationContext) -> List[FitnessScore]:
        """Evaluate multiple ideas"""
        pass
```

### State Store Interface
```python
class IStateStore(ABC):
    @abstractmethod
    async def save_session(self, session: Session) -> None:
        """Persist session state"""
        pass
        
    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[Session]:
        """Load session state"""
        pass
        
    @abstractmethod
    async def list_sessions(self, filter: Optional[Dict[str, Any]] = None) -> List[Session]:
        """List sessions with optional filter"""
        pass
```

## Usage Examples

```python
# Creating a new session
config = SessionConfig(
    population_size=100,
    max_generations=50,
    fitness_weights={
        "relevance": 0.5,
        "novelty": 0.3,
        "feasibility": 0.2
    }
)

session = Session(config=config)

# Generating ideas
task = GenerationTask(
    prompt="Generate innovative solutions for urban transportation",
    temperature=0.9,
    session_id=session.id
)

# Evaluating fitness
context = EvaluationContext(
    target_prompt="sustainable urban mobility",
    existing_ideas=session.populations[-1].ideas if session.populations else []
)

score = await evaluator.evaluate(idea, context)

# Tracking lineage
lineage_node = LineageNode(
    idea_id=idea.id,
    generation=session.current_generation,
    parent_ids=[parent1.id, parent2.id],
    operation="crossover",
    operation_params={"type": "semantic", "split_point": 3}
)
```

These data models provide a comprehensive foundation for implementing the genetic algorithm MCP server with clear contracts and extensibility points.