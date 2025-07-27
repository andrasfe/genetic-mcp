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


class Worker(BaseModel):
    """Represents an LLM worker."""
    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    model: str
    current_task: str | None = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FitnessWeights(BaseModel):
    """Weights for fitness calculation."""
    relevance: float = 0.4
    novelty: float = 0.3
    feasibility: float = 0.3

    @field_validator("relevance", "novelty", "feasibility")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v

    @field_validator("feasibility")
    @classmethod
    def validate_sum(cls, v: float, info) -> float:
        values = info.data
        total = v + values.get("relevance", 0) + values.get("novelty", 0)
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


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
    mode: EvolutionMode = EvolutionMode.SINGLE_PASS
    parameters: GeneticParameters = Field(default_factory=GeneticParameters)
    fitness_weights: FitnessWeights = Field(default_factory=FitnessWeights)
    workers: list[Worker] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    current_generation: int = 0
    status: str = "active"
    client_generated: bool = False  # Whether client generates ideas instead of LLM workers
    ideas_per_generation_received: dict[int, int] = Field(default_factory=dict)  # Track ideas received per generation
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_active_workers(self) -> list[Worker]:
        """Get workers currently working."""
        return [w for w in self.workers if w.status == WorkerStatus.WORKING]

    def get_top_ideas(self, k: int) -> list[Idea]:
        """Get top K ideas by fitness score."""
        sorted_ideas = sorted(self.ideas, key=lambda x: x.fitness, reverse=True)
        return sorted_ideas[:k]


class GenerationRequest(BaseModel):
    """Request to generate ideas."""
    prompt: str
    mode: EvolutionMode = EvolutionMode.SINGLE_PASS
    population_size: int = Field(default=10, ge=2)
    top_k: int = Field(default=5, ge=1)
    generations: int = Field(default=5, ge=1)
    parameters: GeneticParameters | None = None
    fitness_weights: FitnessWeights | None = None
    models: list[str] | None = None  # LLM models to use
    client_generated: bool = Field(default=False)  # Whether client generates ideas instead of LLM workers


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
