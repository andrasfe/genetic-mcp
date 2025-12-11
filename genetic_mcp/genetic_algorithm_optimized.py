"""Optimized genetic algorithm implementation with advanced operators."""

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .advanced_crossover import AdvancedCrossoverManager, CrossoverOperator
from .intelligent_mutation import IntelligentMutationManager
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
        llm_client: LLMClient | None = None,
        advanced_crossover_enabled: bool = False,
        crossover_strategy: str = None,
        crossover_config: dict = None,
        intelligent_mutation_enabled: bool = False,
        target_embedding: list[float] | None = None,
        detail_config: Any | None = None
    ):
        self.parameters = parameters or GeneticParameters()
        self.llm_client = llm_client
        self.metrics = EvolutionMetrics()

        # Adaptive parameters
        self.adaptive_mutation_rate = self.parameters.mutation_rate
        self.adaptive_crossover_rate = self.parameters.crossover_rate
        self.temperature = 1.0  # For Boltzmann selection

        # Advanced crossover setup
        self.advanced_crossover_enabled = advanced_crossover_enabled
        self.crossover_strategy = crossover_strategy
        self.crossover_config = crossover_config or {}

        if self.advanced_crossover_enabled:
            self.crossover_manager = AdvancedCrossoverManager(llm_client=llm_client)
        else:
            self.crossover_manager = None

        # Intelligent mutation setup
        self.intelligent_mutation_enabled = intelligent_mutation_enabled
        self.target_embedding = target_embedding
        self.detail_config = detail_config

        if self.intelligent_mutation_enabled:
            self.mutation_manager = IntelligentMutationManager(llm_client=llm_client)
        else:
            self.mutation_manager = None

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

    async def advanced_crossover(
        self,
        parent1: Idea,
        parent2: Idea,
        generation: int = 0
    ) -> tuple[str, str]:
        """Advanced crossover using the crossover manager."""
        if self.advanced_crossover_enabled and self.crossover_manager:
            # Use advanced crossover manager
            try:
                if self.crossover_strategy:
                    # Use specified strategy
                    operator = CrossoverOperator(self.crossover_strategy)
                else:
                    # Use adaptive selection
                    operator = CrossoverOperator.ADAPTIVE

                offspring1, offspring2 = await self.crossover_manager.crossover(
                    parent1, parent2, operator, generation, **self.crossover_config
                )

                return offspring1, offspring2

            except Exception as e:
                logger.error(f"Advanced crossover failed: {e}")
                # Fallback to semantic crossover
                return await self.semantic_crossover(parent1, parent2)

        else:
            # Use traditional semantic crossover
            return await self.semantic_crossover(parent1, parent2)

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
            # Random temperature between 0.6 and 0.8 for crossover
            temperature = round(random.uniform(0.6, 0.8), 2)
            response = await self.llm_client.generate(
                crossover_prompt,
                temperature=temperature,
                max_tokens=2000
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

    async def batch_crossover(
        self,
        parent_pairs: list[tuple[Idea, Idea]],
        generation: int = 0,
        batch_size: int = 5
    ) -> list[tuple[str, str]]:
        """Perform batch crossover to reduce LLM calls.
        
        Combines multiple crossover operations into fewer LLM calls for efficiency.
        Falls back to individual calls for small batches or when LLM is unavailable.
        """
        if not parent_pairs:
            return []
        
        # For small batches or no LLM, use individual crossovers
        if len(parent_pairs) < 3 or not self.llm_client:
            results = []
            for p1, p2 in parent_pairs:
                result = await self.semantic_crossover(p1, p2)
                results.append(result)
            return results
        
        # Process in batches
        results = []
        for i in range(0, len(parent_pairs), batch_size):
            batch = parent_pairs[i:i + batch_size]
            
            if len(batch) >= 3:
                # Use batch LLM call
                batch_results = await self._batch_crossover_llm(batch)
                results.extend(batch_results)
            else:
                # Individual calls for small remaining batch
                for p1, p2 in batch:
                    result = await self.semantic_crossover(p1, p2)
                    results.append(result)
        
        return results
    
    async def _batch_crossover_llm(
        self,
        parent_pairs: list[tuple[Idea, Idea]]
    ) -> list[tuple[str, str]]:
        """Execute batch crossover using a single LLM call."""
        # Build combined prompt
        prompt_parts = ["You are performing genetic algorithm crossover on multiple idea pairs.\n"]
        prompt_parts.append("For each pair, create two offspring by combining concepts from both parents.\n\n")
        
        for idx, (p1, p2) in enumerate(parent_pairs):
            # Truncate long content to fit in context
            p1_content = p1.content[:500] + "..." if len(p1.content) > 500 else p1.content
            p2_content = p2.content[:500] + "..." if len(p2.content) > 500 else p2.content
            
            prompt_parts.append(f"PAIR_{idx}:\n")
            prompt_parts.append(f"Parent_A: {p1_content}\n")
            prompt_parts.append(f"Parent_B: {p2_content}\n\n")
        
        prompt_parts.append("\nFor each pair, respond with:\n")
        prompt_parts.append("PAIR_X_OFFSPRING_1: [content emphasizing Parent_A concepts]\n")
        prompt_parts.append("PAIR_X_OFFSPRING_2: [content emphasizing Parent_B concepts]\n")
        
        combined_prompt = "".join(prompt_parts)
        
        try:
            temperature = round(random.uniform(0.6, 0.8), 2)
            response = await self.llm_client.generate(
                combined_prompt,
                temperature=temperature,
                max_tokens=3000
            )
            
            # Parse batch response
            results = self._parse_batch_crossover_response(response, len(parent_pairs))
            
            # If parsing failed for some pairs, fill with simple crossover
            while len(results) < len(parent_pairs):
                idx = len(results)
                p1, p2 = parent_pairs[idx]
                results.append(self._simple_crossover(p1, p2))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch crossover LLM call failed: {e}")
            # Fallback to simple crossover for all pairs
            return [self._simple_crossover(p1, p2) for p1, p2 in parent_pairs]
    
    def _parse_batch_crossover_response(
        self,
        response: str,
        expected_pairs: int
    ) -> list[tuple[str, str]]:
        """Parse the batch crossover response from LLM."""
        results = []
        lines = response.strip().split('\n')
        
        current_pair_offspring = {}
        
        for line in lines:
            line = line.strip()
            
            # Try to match PAIR_X_OFFSPRING_Y pattern
            for pair_idx in range(expected_pairs):
                if line.startswith(f"PAIR_{pair_idx}_OFFSPRING_1:"):
                    content = line.replace(f"PAIR_{pair_idx}_OFFSPRING_1:", "").strip()
                    if pair_idx not in current_pair_offspring:
                        current_pair_offspring[pair_idx] = {}
                    current_pair_offspring[pair_idx]['offspring1'] = content
                elif line.startswith(f"PAIR_{pair_idx}_OFFSPRING_2:"):
                    content = line.replace(f"PAIR_{pair_idx}_OFFSPRING_2:", "").strip()
                    if pair_idx not in current_pair_offspring:
                        current_pair_offspring[pair_idx] = {}
                    current_pair_offspring[pair_idx]['offspring2'] = content
        
        # Collect results in order
        for pair_idx in range(expected_pairs):
            if pair_idx in current_pair_offspring:
                pair_data = current_pair_offspring[pair_idx]
                offspring1 = pair_data.get('offspring1', '')
                offspring2 = pair_data.get('offspring2', '')
                if offspring1 and offspring2:
                    results.append((offspring1, offspring2))
        
        return results

    async def adaptive_mutation(self, content: str, generation: int, idea: Idea | None = None, all_ideas: list[Idea] | None = None) -> str:
        """Adaptive mutation with intelligent strategies when enabled."""
        # Use intelligent mutation if enabled and idea is provided
        if self.intelligent_mutation_enabled and self.mutation_manager and idea:
            return await self.intelligent_mutation(idea, all_ideas or [], generation)

        # Fallback to original adaptive mutation
        return await self._legacy_adaptive_mutation(content, generation)

    async def intelligent_mutation(self, idea: Idea, all_ideas: list[Idea], generation: int) -> str:
        """Apply intelligent mutation using the mutation manager."""
        if not self.mutation_manager:
            return await self._legacy_adaptive_mutation(idea.content, generation)

        # Adapt mutation rate based on metrics
        self._update_adaptive_rates()

        if random.random() > self.adaptive_mutation_rate:
            return idea.content

        try:
            mutated_content = await self.mutation_manager.mutate(
                idea=idea,
                all_ideas=all_ideas,
                generation=generation,
                target_embedding=self.target_embedding,
                detail_config=self.detail_config
            )

            logger.debug(f"Intelligent mutation applied to idea {idea.id} in generation {generation}")
            return mutated_content

        except Exception as e:
            logger.error(f"Intelligent mutation failed for idea {idea.id}: {e}")
            # Fallback to legacy mutation
            return await self._legacy_adaptive_mutation(idea.content, generation)

    async def batch_mutation(
        self,
        ideas: list[Idea],
        generation: int,
        all_ideas: list[Idea] | None = None,
        batch_size: int = 5
    ) -> list[str]:
        """Perform batch mutation to reduce LLM calls.
        
        Combines multiple mutation operations into fewer LLM calls.
        """
        if not ideas:
            return []
        
        # Determine which ideas should mutate based on mutation rate
        self._update_adaptive_rates()
        
        to_mutate = []
        no_mutate = []
        mutation_indices = []
        
        for idx, idea in enumerate(ideas):
            if random.random() <= self.adaptive_mutation_rate:
                to_mutate.append(idea)
                mutation_indices.append(idx)
            else:
                no_mutate.append((idx, idea.content))
        
        # If few mutations, use individual calls
        if len(to_mutate) < 3 or not self.llm_client:
            results = [None] * len(ideas)
            for idx, idea in enumerate(ideas):
                mutated = await self.adaptive_mutation(
                    idea.content, generation, idea, all_ideas
                )
                results[idx] = mutated
            return results
        
        # Process mutations in batches
        mutated_contents = []
        for i in range(0, len(to_mutate), batch_size):
            batch = to_mutate[i:i + batch_size]
            
            if len(batch) >= 3:
                batch_results = await self._batch_mutation_llm(batch, generation)
                mutated_contents.extend(batch_results)
            else:
                for idea in batch:
                    mutated = await self.adaptive_mutation(
                        idea.content, generation, idea, all_ideas
                    )
                    mutated_contents.append(mutated)
        
        # Reassemble results in original order
        results = [None] * len(ideas)
        
        # Add non-mutated ideas
        for idx, content in no_mutate:
            results[idx] = content
        
        # Add mutated ideas
        for i, idx in enumerate(mutation_indices):
            if i < len(mutated_contents):
                results[idx] = mutated_contents[i]
            else:
                results[idx] = ideas[idx].content
        
        return results
    
    async def _batch_mutation_llm(
        self,
        ideas: list[Idea],
        generation: int
    ) -> list[str]:
        """Execute batch mutation using a single LLM call."""
        # Choose mutation type based on generation
        if generation < 3:
            mutation_type = "exploratory and creative"
        elif self.metrics.stagnation_counter > 3:
            mutation_type = "disruptive and innovative"
        else:
            mutation_type = "refinement and improvement"
        
        prompt_parts = [
            f"Apply {mutation_type} mutations to the following ideas.\n",
            "Each mutation should modify the idea while preserving its core value.\n\n"
        ]
        
        for idx, idea in enumerate(ideas):
            content = idea.content[:400] + "..." if len(idea.content) > 400 else idea.content
            prompt_parts.append(f"IDEA_{idx}: {content}\n")
        
        prompt_parts.append("\nRespond with mutated versions:\n")
        prompt_parts.append("MUTATED_0: [mutated content]\n")
        prompt_parts.append("MUTATED_1: [mutated content]\n")
        prompt_parts.append("etc.\n")
        
        combined_prompt = "".join(prompt_parts)
        
        try:
            temperature = round(random.uniform(0.7, 0.9), 2)
            response = await self.llm_client.generate(
                combined_prompt,
                temperature=temperature,
                max_tokens=2500
            )
            
            # Parse response
            results = []
            lines = response.strip().split('\n')
            
            for idx in range(len(ideas)):
                found = False
                for line in lines:
                    if line.strip().startswith(f"MUTATED_{idx}:"):
                        content = line.replace(f"MUTATED_{idx}:", "").strip()
                        if content:
                            results.append(content)
                            found = True
                            break
                if not found:
                    results.append(ideas[idx].content)  # No mutation found, keep original
            
            return results
            
        except Exception as e:
            logger.error(f"Batch mutation LLM call failed: {e}")
            return [idea.content for idea in ideas]

    async def _legacy_adaptive_mutation(self, content: str, generation: int) -> str:
        """Legacy adaptive mutation with multiple operators."""
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
            # Variable temperature based on operator with random variation
            if operator == "creative":
                temperature = round(random.uniform(0.75, 0.9), 2)  # Higher for creativity
            elif operator == "disruptive":
                temperature = round(random.uniform(0.8, 0.95), 2)  # Even higher for disruption
            else:
                temperature = round(random.uniform(0.55, 0.7), 2)  # Lower for refinement

            response = await self.llm_client.generate(
                prompts.get(operator, prompts["semantic"]),
                temperature=temperature,
                max_tokens=2000
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

            # Advanced crossover (falls back to semantic if not enabled)
            crossover_result = await self.advanced_crossover(parent1, parent2, generation)
            if isinstance(crossover_result, tuple) and len(crossover_result) == 3:
                # Extended result with operator info
                offspring1_content, offspring2_content, crossover_operator = crossover_result
            else:
                # Standard result
                offspring1_content, offspring2_content = crossover_result
                crossover_operator = "semantic"  # Default

            # Create initial offspring ideas for mutation
            offspring1 = Idea(
                id=f"gen{generation}_offspring{offspring_count}",
                content=offspring1_content,
                generation=generation,
                parent_ids=[parent1.id, parent2.id],
                fitness=0.0,  # Will be evaluated later
                metadata={
                    "selection_method": selection_method,
                    "crossover_operator": crossover_operator
                }
            )

            offspring2 = Idea(
                id=f"gen{generation}_offspring{offspring_count + 1}",
                content=offspring2_content,
                generation=generation,
                parent_ids=[parent1.id, parent2.id],
                fitness=0.0,  # Will be evaluated later
                metadata={
                    "selection_method": selection_method,
                    "crossover_operator": crossover_operator
                }
            )

            # Apply intelligent or adaptive mutation
            offspring1.content = await self.adaptive_mutation(
                offspring1_content, generation, offspring1, population
            )

            if len(new_population) + 1 < self.parameters.population_size:
                offspring2.content = await self.adaptive_mutation(
                    offspring2_content, generation, offspring2, population
                )

            new_population.append(offspring1)
            offspring_count += 1

            if len(new_population) < self.parameters.population_size:
                new_population.append(offspring2)
                offspring_count += 1

        return new_population[:self.parameters.population_size]

    def record_crossover_performance(self, parent1: Idea, parent2: Idea, offspring: Idea):
        """Record crossover performance for adaptive improvement."""
        if self.advanced_crossover_enabled and self.crossover_manager:
            # Extract crossover operator used (if stored in metadata)
            operator = offspring.metadata.get('crossover_operator', 'unknown')
            if operator != 'unknown':
                (parent1.fitness + parent2.fitness) / 2
                self.crossover_manager.record_fitness_improvement(
                    operator, parent1.fitness, parent2.fitness, offspring.fitness
                )

    def get_crossover_performance_report(self) -> dict:
        """Get performance report for crossover operators."""
        if self.advanced_crossover_enabled and self.crossover_manager:
            return self.crossover_manager.get_performance_report()
        return {"message": "Advanced crossover not enabled"}

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

    def update_mutation_feedback(self, population: list[Idea]) -> None:
        """Update mutation manager with fitness feedback for learning."""
        if not self.intelligent_mutation_enabled or not self.mutation_manager:
            return

        for idea in population:
            self.mutation_manager.update_mutation_feedback(idea.id, idea.fitness)

    def get_mutation_performance_report(self) -> dict:
        """Get mutation performance report from intelligent mutation manager."""
        if not self.intelligent_mutation_enabled or not self.mutation_manager:
            return {"message": "Intelligent mutation not enabled"}

        return self.mutation_manager.get_performance_report()

    def reset_mutation_adaptation(self) -> None:
        """Reset mutation adaptation for new session."""
        if self.intelligent_mutation_enabled and self.mutation_manager:
            self.mutation_manager.reset_adaptation()

    def set_target_embedding(self, embedding: list[float]) -> None:
        """Set target embedding for guided mutations."""
        self.target_embedding = embedding

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
