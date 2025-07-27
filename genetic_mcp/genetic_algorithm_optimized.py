"""Optimized genetic algorithm implementation with advanced operators."""

import logging
import random
from dataclasses import dataclass

import numpy as np

from .llm_client import LLMClient
from .models import GeneticParameters, Idea

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetrics:
    """Tracks evolution metrics for adaptive behavior."""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 1.0
    improvement_rate: float = 0.0
    stagnation_counter: int = 0

    def update(self, population: list[Idea]) -> None:
        """Update metrics based on current population."""
        if not population:
            return

        fitnesses = [idea.fitness for idea in population]
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)

        # Simple diversity measure (can be enhanced)
        unique_contents = len(set(idea.content[:50] for idea in population))
        self.diversity = unique_contents / len(population)


class OptimizedGeneticAlgorithm:
    """Optimized genetic algorithm with advanced selection, crossover, and mutation."""

    def __init__(
        self,
        parameters: GeneticParameters = None,
        llm_client: LLMClient | None = None
    ):
        self.parameters = parameters or GeneticParameters()
        self.llm_client = llm_client
        self.metrics = EvolutionMetrics()

        # Adaptive parameters
        self.adaptive_mutation_rate = self.parameters.mutation_rate
        self.adaptive_crossover_rate = self.parameters.crossover_rate
        self.temperature = 1.0  # For Boltzmann selection

    async def select_parents_advanced(
        self,
        population: list[Idea],
        method: str = "tournament"
    ) -> tuple[Idea, Idea]:
        """Advanced parent selection with multiple strategies."""

        if method == "tournament":
            return self._tournament_selection(population)
        elif method == "boltzmann":
            return self._boltzmann_selection(population)
        elif method == "sus":
            return self._stochastic_universal_sampling(population)
        elif method == "rank":
            return self._rank_selection(population)
        else:
            # Fallback to tournament
            return self._tournament_selection(population)

    def _tournament_selection(
        self,
        population: list[Idea],
        tournament_size: int = 3
    ) -> tuple[Idea, Idea]:
        """Tournament selection with adaptive tournament size."""
        # Adapt tournament size based on diversity
        if self.metrics.diversity < 0.3:
            tournament_size = max(2, tournament_size - 1)
        elif self.metrics.diversity > 0.7:
            tournament_size = min(5, tournament_size + 1)

        # Select first parent
        tournament1 = random.sample(population, min(tournament_size, len(population)))
        parent1 = max(tournament1, key=lambda x: x.fitness)

        # Select second parent (different from first)
        remaining = [idea for idea in population if idea.id != parent1.id]
        tournament2 = random.sample(remaining, min(tournament_size, len(remaining)))
        parent2 = max(tournament2, key=lambda x: x.fitness)

        return parent1, parent2

    def _boltzmann_selection(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Boltzmann selection with temperature annealing."""
        # Calculate selection probabilities using Boltzmann distribution
        fitnesses = np.array([idea.fitness for idea in population])

        # Scale fitnesses to prevent overflow
        fitnesses = fitnesses - np.mean(fitnesses)

        # Apply temperature
        probabilities = np.exp(fitnesses / self.temperature)
        probabilities = probabilities / np.sum(probabilities)

        # Select parents
        indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        return population[indices[0]], population[indices[1]]

    def _stochastic_universal_sampling(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Stochastic Universal Sampling for better diversity."""
        fitnesses = [idea.fitness for idea in population]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.sample(population, 2)

        # Calculate cumulative probabilities
        probabilities = [f / total_fitness for f in fitnesses]
        cumulative_probs = np.cumsum(probabilities)

        # Generate two evenly spaced pointers
        start = random.random() / 2
        pointers = [start, start + 0.5]

        # Select individuals
        selected = []
        for pointer in pointers:
            for i, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    selected.append(population[i])
                    break

        return selected[0], selected[1] if len(selected) > 1 else selected[0]

    def _rank_selection(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Rank-based selection to reduce selection pressure."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness)

        # Assign rank-based probabilities
        n = len(sorted_pop)
        ranks = list(range(1, n + 1))
        total_ranks = sum(ranks)
        probabilities = [rank / total_ranks for rank in ranks]

        # Select parents
        indices = np.random.choice(n, size=2, replace=False, p=probabilities)
        return sorted_pop[indices[0]], sorted_pop[indices[1]]

    async def semantic_crossover(
        self,
        parent1: Idea,
        parent2: Idea
    ) -> tuple[str, str]:
        """LLM-guided semantic crossover."""
        if not self.llm_client or random.random() > self.adaptive_crossover_rate:
            # Fallback to simple crossover
            return self._simple_crossover(parent1, parent2)

        # Use LLM to identify key concepts and blend them
        crossover_prompt = f"""
        You are helping with genetic algorithm crossover of ideas.

        Parent 1: {parent1.content}
        Parent 2: {parent2.content}

        Create two offspring by intelligently combining concepts from both parents:
        1. First offspring should emphasize concepts from Parent 1 with elements from Parent 2
        2. Second offspring should emphasize concepts from Parent 2 with elements from Parent 1

        Format your response as:
        OFFSPRING_1: [content]
        OFFSPRING_2: [content]
        """

        try:
            response = await self.llm_client.generate(
                crossover_prompt,
                temperature=0.7,
                max_tokens=500
            )

            # Parse response
            lines = response.strip().split('\n')
            offspring1 = offspring2 = None

            for line in lines:
                if line.startswith("OFFSPRING_1:"):
                    offspring1 = line.replace("OFFSPRING_1:", "").strip()
                elif line.startswith("OFFSPRING_2:"):
                    offspring2 = line.replace("OFFSPRING_2:", "").strip()

            if offspring1 and offspring2:
                return offspring1, offspring2

        except Exception as e:
            logger.error(f"Semantic crossover failed: {e}")

        # Fallback to simple crossover
        return self._simple_crossover(parent1, parent2)

    def _simple_crossover(self, parent1: Idea, parent2: Idea) -> tuple[str, str]:
        """Simple crossover fallback."""
        # Extract sentences
        sentences1 = [s.strip() for s in parent1.content.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2.content.split('.') if s.strip()]

        if not sentences1 or not sentences2:
            return parent1.content, parent2.content

        # Crossover at midpoint
        mid1 = len(sentences1) // 2
        mid2 = len(sentences2) // 2

        offspring1 = '. '.join(sentences1[:mid1] + sentences2[mid2:]) + '.'
        offspring2 = '. '.join(sentences2[:mid2] + sentences1[mid1:]) + '.'

        return offspring1, offspring2

    async def adaptive_mutation(self, content: str, generation: int) -> str:
        """Adaptive mutation with multiple operators."""
        # Adapt mutation rate based on metrics
        self._update_adaptive_rates()

        if random.random() > self.adaptive_mutation_rate:
            return content

        # Choose mutation operator based on evolution stage
        if generation < 3:
            # Early generations: more exploratory
            operators = ["semantic", "creative", "expand"]
        elif self.metrics.stagnation_counter > 3:
            # Stagnation: more disruptive
            operators = ["disruptive", "hybrid", "creative"]
        else:
            # Normal evolution
            operators = ["semantic", "refine", "modify"]

        operator = random.choice(operators)

        if self.llm_client and operator in ["semantic", "creative", "disruptive"]:
            return await self._llm_mutation(content, operator)
        else:
            return self._basic_mutation(content, operator)

    async def _llm_mutation(self, content: str, operator: str) -> str:
        """LLM-based mutation operators."""
        prompts = {
            "semantic": f"Slightly modify this idea while keeping its core concept: {content}",
            "creative": f"Add a creative twist or unexpected element to this idea: {content}",
            "disruptive": f"Significantly transform this idea into something related but different: {content}"
        }

        try:
            response = await self.llm_client.generate(
                prompts.get(operator, prompts["semantic"]),
                temperature=0.8 if operator == "creative" else 0.6,
                max_tokens=200
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM mutation failed: {e}")
            return self._basic_mutation(content, "modify")

    def _basic_mutation(self, content: str, operator: str) -> str:
        """Basic mutation operators."""
        if operator == "expand":
            additions = [
                " This could be enhanced with AI/ML techniques.",
                " Consider implementing this in phases for better adoption.",
                " User testing would provide valuable insights.",
                " This approach offers scalability advantages."
            ]
            return content + random.choice(additions)

        elif operator == "refine":
            # Simple refinements
            replacements = [
                ("could", "should"),
                ("might", "will"),
                ("basic", "comprehensive"),
                ("simple", "sophisticated")
            ]
            result = content
            for old, new in replacements:
                if old in result:
                    result = result.replace(old, new, 1)
                    break
            return result

        elif operator == "hybrid":
            # Combine multiple simple mutations
            result = content
            if random.random() < 0.5:
                result = self._basic_mutation(result, "expand")
            if random.random() < 0.5:
                result = self._basic_mutation(result, "refine")
            return result

        else:  # modify
            sentences = content.split('.')
            if len(sentences) > 2:
                # Shuffle middle sentences
                middle = sentences[1:-1]
                random.shuffle(middle)
                return sentences[0] + '. ' + '. '.join(middle) + '. ' + sentences[-1]
            return content

    def _update_adaptive_rates(self) -> None:
        """Update adaptive parameters based on evolution metrics."""
        # Increase mutation if diversity is low
        if self.metrics.diversity < 0.3:
            self.adaptive_mutation_rate = min(0.5, self.parameters.mutation_rate * 1.5)
        elif self.metrics.diversity > 0.7:
            self.adaptive_mutation_rate = max(0.05, self.parameters.mutation_rate * 0.7)
        else:
            self.adaptive_mutation_rate = self.parameters.mutation_rate

        # Adjust crossover based on improvement
        if self.metrics.improvement_rate < 0.01:
            self.adaptive_crossover_rate = min(0.9, self.parameters.crossover_rate * 1.2)
        else:
            self.adaptive_crossover_rate = self.parameters.crossover_rate

        # Temperature annealing for Boltzmann selection
        self.temperature = max(0.1, self.temperature * 0.95)

    async def create_next_generation_optimized(
        self,
        population: list[Idea],
        generation: int,
        selection_method: str = "tournament"
    ) -> list[Idea]:
        """Create next generation with optimized operators."""
        new_population = []

        # Update metrics
        self.metrics.generation = generation
        self.metrics.update(population)

        # Elitism with adaptive count
        elite_count = self._adaptive_elitism_count(population)
        if elite_count > 0:
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            elite = sorted_pop[:elite_count]

            for idx, elite_idea in enumerate(elite):
                new_idea = Idea(
                    id=f"gen{generation}_elite{idx}",
                    content=elite_idea.content,
                    generation=generation,
                    parent_ids=[elite_idea.id],
                    metadata={"elite": True, "preserved_fitness": elite_idea.fitness}
                )
                new_population.append(new_idea)

        # Generate offspring
        offspring_count = 0
        while len(new_population) < self.parameters.population_size:
            # Select parents with chosen method
            parent1, parent2 = await self.select_parents_advanced(population, selection_method)

            # Semantic crossover
            offspring1_content, offspring2_content = await self.semantic_crossover(parent1, parent2)

            # Adaptive mutation
            offspring1_content = await self.adaptive_mutation(offspring1_content, generation)
            offspring2_content = await self.adaptive_mutation(offspring2_content, generation)

            # Create offspring
            offspring1 = Idea(
                id=f"gen{generation}_offspring{offspring_count}",
                content=offspring1_content,
                generation=generation,
                parent_ids=[parent1.id, parent2.id],
                metadata={"selection_method": selection_method}
            )
            new_population.append(offspring1)
            offspring_count += 1

            if len(new_population) < self.parameters.population_size:
                offspring2 = Idea(
                    id=f"gen{generation}_offspring{offspring_count}",
                    content=offspring2_content,
                    generation=generation,
                    parent_ids=[parent1.id, parent2.id],
                    metadata={"selection_method": selection_method}
                )
                new_population.append(offspring2)
                offspring_count += 1

        return new_population[:self.parameters.population_size]

    def _adaptive_elitism_count(self, population: list[Idea]) -> int:
        """Adaptively determine elitism count."""
        base_elite = self.parameters.elitism_count

        # Increase elitism if converging well
        if self.metrics.improvement_rate > 0.1:
            return min(base_elite + 1, len(population) // 4)

        # Decrease elitism if stagnating
        if self.metrics.stagnation_counter > 3:
            return max(1, base_elite - 1)

        return base_elite

    def check_convergence(self, population: list[Idea], threshold: float = 0.01) -> bool:
        """Check if population has converged."""
        if len(population) < 2:
            return False

        fitnesses = [idea.fitness for idea in population]
        fitness_variance = np.var(fitnesses)

        # Check improvement rate
        if hasattr(self, '_last_best_fitness'):
            improvement = self.metrics.best_fitness - self._last_best_fitness
            self.metrics.improvement_rate = improvement

            if improvement < threshold:
                self.metrics.stagnation_counter += 1
            else:
                self.metrics.stagnation_counter = 0

        self._last_best_fitness = self.metrics.best_fitness

        # Converged if low variance and high stagnation
        return fitness_variance < 0.01 and self.metrics.stagnation_counter > 5
