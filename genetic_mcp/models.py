"""Data models for the Genetic MCP server."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class WorkerStatus(str, Enum):
    """Status of an LLM worker."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class EvolutionMode(str, Enum):
    """Evolution mode for genetic algorithm."""
    SINGLE_PASS = "single_pass"  # Just rank top-K
    ITERATIVE = "iterative"  # Full genetic algorithm


class Idea(BaseModel):
    """Represents a generated idea."""
    id: str
    content: str
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # relevance, novelty, feasibility
    fitness: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Claude evaluation fields
    claude_evaluation: dict[str, Any] | None = None  # Structured evaluation from Claude
    claude_score: float | None = None  # Claude's overall score (0-1)
    combined_fitness: float | None = None  # Combined algorithmic + Claude fitness


class Worker(BaseModel):
    """Represents an LLM worker."""
    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    model: str
    current_task: str | None = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DetailConfig(BaseModel):
    """Configuration for detail level in generated ideas."""
    level: str = Field(default="medium")  # "low", "medium", "high"
    require_code_examples: bool = False
    require_step_by_step: bool = False
    require_technical_specs: bool = False
    min_sections: int = Field(default=0, ge=0)

    def get_detail_prompt_fragment(self) -> str:
        """Generate prompt fragment based on detail configuration."""
        fragments = []

        # Base detail level instruction
        if self.level == "high":
            fragments.append("Provide comprehensive, detailed responses with thorough explanations.")
        elif self.level == "medium":
            fragments.append("Provide balanced responses with sufficient detail.")
        elif self.level == "low":
            fragments.append("Provide concise, focused responses.")

        # Specific requirements
        if self.require_code_examples:
            fragments.append(
                "Include specific code examples with concrete implementations. "
                "Show actual syntax and working code snippets that demonstrate the concept."
            )

        if self.require_step_by_step:
            fragments.append(
                "Break down the solution into numbered, actionable steps. "
                "Each step should be clear, specific, and implementable."
            )

        if self.require_technical_specs:
            fragments.append(
                "Include technical specifications such as: "
                "specific technologies/frameworks to use, architecture patterns, "
                "data structures, algorithms, API designs, and performance considerations."
            )

        if self.min_sections > 0:
            fragments.append(
                f"Structure your response with at least {self.min_sections} distinct sections "
                "covering different aspects of the solution."
            )

        return " ".join(fragments)


class FitnessWeights(BaseModel):
    """Weights for fitness calculation.

    Base weights (relevance, novelty, feasibility) must sum to 1.0.
    Detail-aware weights are optional and applied within feasibility calculation.
    """
    relevance: float = 0.4
    novelty: float = 0.3
    feasibility: float = 0.3

    # Optional detail-aware metric weights (used within feasibility if provided)
    # These represent the composition of the detail score, should sum to 1.0 if used
    implementation_depth: float = 0.0  # δ (delta)
    actionability: float = 0.0  # α (alpha)
    completeness: float = 0.0  # κ (kappa)
    technical_precision: float = 0.0  # τ (tau)

    @field_validator("relevance", "novelty", "feasibility",
                    "implementation_depth", "actionability", "completeness", "technical_precision")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v

    @field_validator("feasibility")
    @classmethod
    def validate_base_sum(cls, v: float, info) -> float:
        """Validate that base weights sum to 1.0."""
        values = info.data
        total = v + values.get("relevance", 0) + values.get("novelty", 0)
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Base weights (relevance, novelty, feasibility) must sum to 1.0, got {total}")
        return v

    @field_validator("technical_precision")
    @classmethod
    def validate_detail_sum(cls, v: float, info) -> float:
        """Validate that detail weights sum to 1.0 if any are non-zero."""
        values = info.data
        detail_weights = [
            values.get("implementation_depth", 0),
            values.get("actionability", 0),
            values.get("completeness", 0),
            v  # technical_precision
        ]

        total = sum(detail_weights)

        # If any detail weights are provided, they should sum to 1.0
        if total > 0.001 and abs(total - 1.0) > 0.001:  # Some detail weights provided
            raise ValueError(
                f"Detail weights (implementation_depth, actionability, completeness, technical_precision) "
                f"must sum to 1.0 when used, got {total}"
            )

        return v

    def has_detail_weights(self) -> bool:
        """Check if detail-aware weights are configured."""
        return (self.implementation_depth > 0 or self.actionability > 0 or
                self.completeness > 0 or self.technical_precision > 0)


class GeneticParameters(BaseModel):
    """Parameters for genetic algorithm."""
    population_size: int = Field(default=10, ge=2)
    generations: int = Field(default=5, ge=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    crossover_rate: float = Field(default=0.7, ge=0, le=1)
    elitism_count: int = Field(default=2, ge=0)

    @field_validator("elitism_count")
    @classmethod
    def validate_elitism(cls, v: int, info) -> int:
        values = info.data
        pop_size = values.get("population_size", 10)
        if v > pop_size // 2:
            raise ValueError(f"Elitism count ({v}) must be at most half of population size ({pop_size})")
        return v


class Session(BaseModel):
    """Represents a generation session."""
    id: str
    client_id: str
    prompt: str
    mode: EvolutionMode = EvolutionMode.ITERATIVE  # Changed from SINGLE_PASS to enable genetic algorithm by default
    parameters: GeneticParameters = Field(default_factory=GeneticParameters)
    fitness_weights: FitnessWeights = Field(default_factory=FitnessWeights)
    detail_config: DetailConfig = Field(default_factory=DetailConfig)
    workers: list[Worker] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    current_generation: int = 0
    status: str = "active"
    client_generated: bool = False  # Whether client generates ideas instead of LLM workers
    claude_evaluation_enabled: bool = False  # Whether Claude assists with evaluation
    claude_evaluation_weight: float = 0.5  # Weight for Claude's evaluation (0-1)
    ideas_per_generation_received: dict[int, int] = Field(default_factory=dict)  # Track ideas received per generation
    adaptive_population_enabled: bool = False  # Whether adaptive population size is enabled
    adaptive_population_config: dict[str, Any] = Field(default_factory=dict)  # Config for adaptive population
    memory_enabled: bool = False  # Whether to use memory system for parameter optimization
    parameter_recommendation: dict[str, Any] = Field(default_factory=dict)  # Store parameter recommendation used
    execution_time_seconds: float = 0.0  # Track execution time for memory learning

    # Hybrid selection strategy configuration
    hybrid_selection_enabled: bool = False  # Whether to use hybrid selection strategies
    selection_strategy: str | None = None  # Manual strategy override (e.g., "tournament", "roulette_wheel")
    selection_adaptation_window: int = 5  # Number of generations for performance tracking
    selection_exploration_constant: float = 2.0  # UCB1 exploration parameter
    selection_min_uses_per_strategy: int = 3  # Minimum uses before adaptation
    selection_performance_history: dict[str, Any] = Field(default_factory=dict)  # Strategy performance tracking

    # Advanced crossover configuration
    advanced_crossover_enabled: bool = False  # Whether to use advanced crossover operators
    crossover_strategy: str | None = None  # Manual crossover operator override (e.g., "semantic", "multi_point")
    crossover_adaptation_enabled: bool = True  # Whether to adaptively select crossover operators
    crossover_performance_tracking: bool = True  # Track crossover operator performance
    crossover_performance_history: dict[str, Any] = Field(default_factory=dict)  # Crossover performance tracking
    crossover_config: dict[str, Any] = Field(default_factory=dict)  # Additional crossover configuration

    # Intelligent mutation configuration
    intelligent_mutation_enabled: bool = False  # Whether to use intelligent mutation strategies
    mutation_strategy: str | None = None  # Manual mutation strategy override (e.g., "guided", "adaptive", "memetic")
    mutation_adaptation_enabled: bool = True  # Whether to adaptively learn mutation patterns
    mutation_performance_tracking: bool = True  # Track mutation strategy performance
    mutation_performance_history: dict[str, Any] = Field(default_factory=dict)  # Mutation performance tracking
    mutation_config: dict[str, Any] = Field(default_factory=dict)  # Additional mutation configuration
    target_embedding: list[float] | None = None  # Target embedding for guided mutations

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_active_workers(self) -> list[Worker]:
        """Get workers currently working."""
        return [w for w in self.workers if w.status == WorkerStatus.WORKING]

    def get_top_ideas(self, k: int) -> list[Idea]:
        """Get top K ideas by fitness score."""
        sorted_ideas = sorted(self.ideas, key=lambda x: x.fitness, reverse=True)
        return sorted_ideas[:k]

    def get_current_generation_ideas(self) -> list[Idea]:
        """Get ideas from the current generation."""
        return [idea for idea in self.ideas if idea.generation == self.current_generation]

    def get_population_for_generation(self, generation: int) -> list[Idea]:
        """Get population for a specific generation."""
        return [idea for idea in self.ideas if idea.generation == generation]

    def get_recommended_population_size(self, diversity_metrics: dict[str, float] = None) -> int:
        """Get recommended population size for next generation using adaptive population management."""
        if not self.adaptive_population_enabled or not hasattr(self, '_adaptive_population_manager'):
            return self.parameters.population_size

        # Get current generation population
        current_population = self.get_current_generation_ideas()
        if not current_population:
            current_population = self.ideas[-self.parameters.population_size:] if self.ideas else []

        # Analyze current population
        current_metrics = self._adaptive_population_manager.analyze_population(
            current_population, self.current_generation, diversity_metrics
        )

        # Get recommendation for next generation
        return self._adaptive_population_manager.get_recommended_population_size(
            current_metrics, self.current_generation + 1
        )

    def update_population_size_dynamically(self, new_size: int) -> None:
        """Update population size for next generation."""
        if hasattr(self, '_adaptive_population_manager'):
            self._adaptive_population_manager.update_session_population_size(self, new_size)


class GenerationRequest(BaseModel):
    """Request to generate ideas."""
    prompt: str
    mode: EvolutionMode = EvolutionMode.ITERATIVE  # Changed from SINGLE_PASS to enable genetic algorithm by default
    population_size: int = Field(default=10, ge=2)
    top_k: int = Field(default=5, ge=1)
    generations: int = Field(default=5, ge=1)
    parameters: GeneticParameters | None = None
    fitness_weights: FitnessWeights | None = None
    detail_config: DetailConfig | None = None  # Configuration for detail level in generated ideas
    models: list[str] | None = None  # LLM models to use
    client_generated: bool = Field(default=False)  # Whether client generates ideas instead of LLM workers
    adaptive_population: bool = Field(default=False)  # Enable adaptive population size
    adaptive_population_config: dict[str, Any] | None = None  # Configuration for adaptive population
    use_memory_system: bool = Field(default=True)  # Whether to use memory system for parameter optimization
    optimization_level: str | None = None  # Optimization level: 'low', 'medium', 'high', 'auto'

    # Hybrid selection strategy options
    hybrid_selection_enabled: bool = Field(default=False)  # Enable hybrid selection strategies
    selection_strategy: str | None = None  # Manual strategy override
    selection_adaptation_window: int = Field(default=5, ge=1)  # Performance tracking window
    selection_exploration_constant: float = Field(default=2.0, ge=0.1)  # UCB1 exploration parameter

    # Advanced crossover options
    advanced_crossover_enabled: bool = Field(default=False)  # Enable advanced crossover operators
    crossover_strategy: str | None = None  # Manual crossover operator override
    crossover_adaptation_enabled: bool = Field(default=True)  # Enable adaptive crossover selection
    crossover_config: dict[str, Any] | None = None  # Additional crossover configuration

    # Intelligent mutation options
    intelligent_mutation_enabled: bool = Field(default=False)  # Enable intelligent mutation strategies
    mutation_strategy: str | None = None  # Manual mutation strategy override
    mutation_adaptation_enabled: bool = Field(default=True)  # Enable adaptive mutation learning
    mutation_config: dict[str, Any] | None = None  # Additional mutation configuration


class GenerationProgress(BaseModel):
    """Progress update during generation."""
    session_id: str
    current_generation: int
    total_generations: int
    ideas_generated: int
    active_workers: int
    best_fitness: float
    status: str
    message: str | None = None


class GenerationResult(BaseModel):
    """Result of idea generation."""
    session_id: str
    top_ideas: list[Idea]
    total_ideas_generated: int
    generations_completed: int
    lineage: dict[str, list[str]] = Field(default_factory=dict)  # idea_id -> parent_ids
    execution_time_seconds: float


class ClaudeEvaluationRequest(BaseModel):
    """Request for Claude to evaluate ideas."""
    session_id: str
    ideas: list[Idea]  # Ideas to evaluate
    prompt: str  # Original prompt for context
    evaluation_criteria: dict[str, str] | None = None  # Custom evaluation criteria
    batch_size: int = Field(default=10, ge=1, le=50)  # Max ideas per evaluation


class ClaudeEvaluationResponse(BaseModel):
    """Response from Claude's evaluation."""
    session_id: str
    evaluations: dict[str, dict[str, Any]]  # idea_id -> evaluation
    evaluation_time_seconds: float
