"""Genetic algorithm implementation for idea evolution."""

import logging
import random
import re

from .models import GeneticParameters, Idea

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Implements genetic algorithm operations for idea evolution."""

    def __init__(self, parameters: GeneticParameters | None = None):
        self.parameters = parameters or GeneticParameters()

    def select_parents(self, population: list[Idea], probabilities: list[float]) -> tuple[Idea, Idea]:
        """Select two parents for crossover using roulette wheel selection."""
        if len(population) < 2:
            raise ValueError("Population must have at least 2 ideas for selection")

        # Select first parent
        parent1 = random.choices(population, weights=probabilities, k=1)[0]

        # Select second parent (different from first)
        remaining_pop = [idea for idea in population if idea.id != parent1.id]
        remaining_probs = [prob for idea, prob in zip(population, probabilities, strict=False) if idea.id != parent1.id]

        # Normalize remaining probabilities
        total_prob = sum(remaining_probs)
        if total_prob > 0:
            remaining_probs = [p / total_prob for p in remaining_probs]
        else:
            remaining_probs = [1.0 / len(remaining_pop)] * len(remaining_pop)

        parent2 = random.choices(remaining_pop, weights=remaining_probs, k=1)[0]

        return parent1, parent2

    def crossover(self, parent1: Idea, parent2: Idea) -> tuple[str, str]:
        """Perform crossover between two parent ideas."""
        if random.random() > self.parameters.crossover_rate:
            # No crossover, return parents as-is
            return parent1.content, parent2.content

        # Extract sentences or key points from both parents
        sentences1 = self._extract_sentences(parent1.content)
        sentences2 = self._extract_sentences(parent2.content)

        if not sentences1 or not sentences2:
            return parent1.content, parent2.content

        # Perform crossover at sentence level
        crossover_point1 = random.randint(1, len(sentences1) - 1) if len(sentences1) > 1 else 1
        crossover_point2 = random.randint(1, len(sentences2) - 1) if len(sentences2) > 1 else 1

        # Create offspring
        offspring1_parts = sentences1[:crossover_point1] + sentences2[crossover_point2:]
        offspring2_parts = sentences2[:crossover_point2] + sentences1[crossover_point1:]

        offspring1 = " ".join(offspring1_parts)
        offspring2 = " ".join(offspring2_parts)

        return offspring1, offspring2

    def mutate(self, content: str) -> str:
        """Apply mutation to idea content."""
        if random.random() > self.parameters.mutation_rate:
            return content

        # Try mutations until we get a change (important for 100% mutation rate)
        original_content = content
        max_attempts = 5

        for _ in range(max_attempts):
            mutation_type = random.choice(["rephrase", "add", "remove", "modify"])

            if mutation_type == "rephrase":
                mutated = self._rephrase_mutation(content)
            elif mutation_type == "add":
                mutated = self._add_mutation(content)
            elif mutation_type == "remove":
                mutated = self._remove_mutation(content)
            else:  # modify
                mutated = self._modify_mutation(content)

            # If mutation produced a change, return it
            if mutated != original_content:
                return mutated

        # If no mutation produced a change, force a simple change
        return f"{content} (mutated)"

    def _extract_sentences(self, content: str) -> list[str]:
        """Extract sentences or key points from content."""
        # Try to split by sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            # Try splitting by newlines or bullet points
            sentences = re.split(r'[\n•\-*]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _rephrase_mutation(self, content: str) -> str:
        """Rephrase part of the content."""
        sentences = self._extract_sentences(content)
        if not sentences:
            return content

        # Select random sentence to modify
        idx = random.randint(0, len(sentences) - 1)
        sentence = sentences[idx]

        # Simple rephrasing by adding variation phrases
        variations = [
            f"Additionally, {sentence.lower()}",
            f"Furthermore, {sentence.lower()}",
            f"It's worth noting that {sentence.lower()}",
            f"Importantly, {sentence.lower()}",
            f"{sentence} This is crucial because it enables further innovation.",
            f"{sentence} This approach offers significant advantages."
        ]

        sentences[idx] = random.choice(variations)
        return " ".join(sentences)

    def _add_mutation(self, content: str) -> str:
        """Add new element to content."""
        additions = [
            "Consider the scalability implications of this approach.",
            "This could be enhanced with machine learning techniques.",
            "User feedback would be essential for validation.",
            "Performance optimization should be a key consideration.",
            "Security aspects need careful attention.",
            "This aligns with current industry best practices."
        ]

        return f"{content} {random.choice(additions)}"

    def _remove_mutation(self, content: str) -> str:
        """Remove element from content."""
        sentences = self._extract_sentences(content)
        if len(sentences) <= 1:
            return content

        # Remove random sentence
        idx = random.randint(0, len(sentences) - 1)
        sentences.pop(idx)
        return " ".join(sentences)

    def _modify_mutation(self, content: str) -> str:
        """Modify specific aspects of content."""
        modifications = [
            ("small", "large"),
            ("simple", "complex"),
            ("basic", "advanced"),
            ("traditional", "innovative"),
            ("sequential", "parallel"),
            ("manual", "automated")
        ]

        modified = content
        for old, new in modifications:
            if old in modified.lower():
                # Replace with proper case handling
                modified = re.sub(
                    rf'\b{old}\b',
                    new,
                    modified,
                    flags=re.IGNORECASE
                )
                break

        return modified

    def create_next_generation(self, population: list[Idea],
                             probabilities: list[float],
                             generation: int) -> list[Idea]:
        """Create next generation using genetic operations."""
        new_population: list[Idea] = []

        # Elitism: Keep top performers
        if self.parameters.elitism_count > 0:
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            elite = sorted_pop[:self.parameters.elitism_count]

            # Create new ideas from elite (they survive unchanged)
            for idx, elite_idea in enumerate(elite):
                new_idea = Idea(
                    id=f"gen{generation}_elite{idx}",
                    content=elite_idea.content,
                    generation=generation,
                    parent_ids=[elite_idea.id],
                    metadata={"elite": True}
                )
                new_population.append(new_idea)

        # Generate rest of population through crossover and mutation
        while len(new_population) < self.parameters.population_size:
            # Select parents
            parent1, parent2 = self.select_parents(population, probabilities)

            # Perform crossover
            offspring1_content, offspring2_content = self.crossover(parent1, parent2)

            # Apply mutation
            offspring1_content = self.mutate(offspring1_content)
            offspring2_content = self.mutate(offspring2_content)

            # Create new ideas
            offspring1 = Idea(
                id=f"gen{generation}_offspring{len(new_population)}",
                content=offspring1_content,
                generation=generation,
                parent_ids=[parent1.id, parent2.id]
            )
            new_population.append(offspring1)

            if len(new_population) < self.parameters.population_size:
                offspring2 = Idea(
                    id=f"gen{generation}_offspring{len(new_population)}",
                    content=offspring2_content,
                    generation=generation,
                    parent_ids=[parent1.id, parent2.id]
                )
                new_population.append(offspring2)

        # Trim to exact population size if needed
        return new_population[:self.parameters.population_size]
