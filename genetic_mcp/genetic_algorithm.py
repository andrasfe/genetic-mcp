"""Genetic algorithm implementation for idea evolution."""

import logging
import random
import re

import numpy as np

from .models import GeneticParameters, Idea

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Implements genetic algorithm operations for idea evolution."""

    def __init__(self, parameters: GeneticParameters | None = None):
        self.parameters = parameters or GeneticParameters()

        # Adaptive parameter tracking
        self.generation_count = 0
        self.fitness_history: list[float] = []
        self.diversity_history: list[float] = []
        self.stagnation_counter = 0

        # Selection strategy parameters
        self.boltzmann_temperature = 1.0
        self.sus_pointers = 2  # Stochastic Universal Sampling pointers
        self.rank_selection_pressure = 2.0  # Linear ranking parameter

    def select_parents(self, population: list[Idea], probabilities: list[float],
                       method: str = "roulette") -> tuple[Idea, Idea]:
        """Select two parents for crossover using specified selection method.

        Args:
            population: List of ideas to select from
            probabilities: Selection probabilities for each idea
            method: Selection method - 'roulette', 'tournament', 'boltzmann', 'sus', 'rank'

        Returns:
            Tuple of two parent ideas
        """
        if method == "tournament":
            return self._tournament_selection(population)
        elif method == "boltzmann":
            return self._boltzmann_selection(population)
        elif method == "sus":
            return self._stochastic_universal_sampling(population)
        elif method == "rank":
            return self._rank_based_selection(population)
        else:
            return self._roulette_wheel_selection(population, probabilities)

    def _roulette_wheel_selection(self, population: list[Idea], probabilities: list[float]) -> tuple[Idea, Idea]:
        """Original roulette wheel selection."""
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

    def _tournament_selection(self, population: list[Idea], tournament_size: int = 3) -> tuple[Idea, Idea]:
        """Tournament selection with configurable tournament size.

        Mathematical basis: Selection pressure = tournament_size
        Probability of selecting best individual = 1 - (1 - 1/N)^k
        where N is population size and k is tournament size.
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 ideas")

        # Adaptive tournament size based on diversity
        if hasattr(self, 'diversity_history') and self.diversity_history:
            current_diversity = self.diversity_history[-1] if self.diversity_history else 0.5
            if current_diversity < 0.3:
                tournament_size = max(2, tournament_size - 1)  # Reduce pressure
            elif current_diversity > 0.7:
                tournament_size = min(5, tournament_size + 1)  # Increase pressure

        # Select first parent
        tournament1 = random.sample(population, min(tournament_size, len(population)))
        parent1 = max(tournament1, key=lambda x: x.fitness)

        # Select second parent (ensuring it's different)
        remaining = [idea for idea in population if idea.id != parent1.id]
        if not remaining:
            remaining = population
        tournament2 = random.sample(remaining, min(tournament_size, len(remaining)))
        parent2 = max(tournament2, key=lambda x: x.fitness)

        return parent1, parent2

    def _boltzmann_selection(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Boltzmann selection with temperature-based probability distribution.

        Mathematical basis: P(i) = exp(f_i/T) / Σ(exp(f_j/T))
        where f_i is fitness of individual i, T is temperature.
        Temperature annealing: T(t) = T_0 * α^t, where α ∈ (0,1)
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 ideas")

        # Extract fitnesses and normalize to prevent numerical overflow
        fitnesses = np.array([idea.fitness for idea in population])

        # Shift fitnesses to prevent negative exponents
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-6

        # Scale fitnesses by temperature
        scaled_fitnesses = fitnesses / self.boltzmann_temperature

        # Prevent overflow by capping maximum scaled fitness
        max_scaled = np.max(scaled_fitnesses)
        if max_scaled > 700:  # exp(700) is near float64 limit
            scaled_fitnesses = scaled_fitnesses * (700 / max_scaled)

        # Calculate Boltzmann probabilities
        exp_fitnesses = np.exp(scaled_fitnesses)
        probabilities = exp_fitnesses / np.sum(exp_fitnesses)

        # Ensure probabilities sum to 1 (handle numerical errors)
        probabilities = probabilities / np.sum(probabilities)

        # Select two different parents
        try:
            indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        except ValueError:
            # Fallback if probabilities are invalid
            indices = np.random.choice(len(population), size=2, replace=False)

        return population[indices[0]], population[indices[1]]

    def _stochastic_universal_sampling(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Stochastic Universal Sampling (SUS) - reduces selection bias.

        Mathematical basis: Places N equally spaced pointers on roulette wheel.
        Ensures even sampling across fitness landscape.
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 ideas")

        # Calculate cumulative fitness
        fitnesses = [max(0, idea.fitness) for idea in population]  # Ensure non-negative
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            # Equal probability if all have zero fitness
            return random.sample(population, 2)

        # Calculate cumulative probabilities
        cumulative_probs = []
        cumsum = 0
        for fitness in fitnesses:
            cumsum += fitness / total_fitness
            cumulative_probs.append(cumsum)

        # Generate evenly spaced pointers
        pointer_distance = 1.0 / self.sus_pointers
        start = random.random() * pointer_distance
        pointers = [start + i * pointer_distance for i in range(self.sus_pointers)]

        # Select individuals
        selected_indices = []
        for pointer in pointers:
            for i, cum_prob in enumerate(cumulative_probs):
                if pointer <= cum_prob:
                    selected_indices.append(i)
                    break

        # Ensure we have at least 2 unique selections
        selected_indices = list(set(selected_indices))
        while len(selected_indices) < 2:
            new_idx = random.randint(0, len(population) - 1)
            if new_idx not in selected_indices:
                selected_indices.append(new_idx)

        return population[selected_indices[0]], population[selected_indices[1]]

    def _rank_based_selection(self, population: list[Idea]) -> tuple[Idea, Idea]:
        """Rank-based selection to reduce effect of fitness scaling.

        Mathematical basis: Linear ranking - P(i) = (2-SP)/N + 2*rank(i)*(SP-1)/(N*(N-1))
        where SP is selection pressure (1 ≤ SP ≤ 2), N is population size.
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 ideas")

        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_pop)

        # Calculate rank-based probabilities using linear ranking
        sp = self.rank_selection_pressure  # Selection pressure
        probabilities = []

        for i in range(n):
            rank = i + 1  # Rank from 1 to n
            # Linear ranking formula
            prob = (2 - sp) / n + 2 * rank * (sp - 1) / (n * (n - 1))
            probabilities.append(max(0, prob))  # Ensure non-negative

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / n] * n

        # Select parents
        try:
            indices = np.random.choice(n, size=2, replace=False, p=probabilities)
        except ValueError:
            indices = np.random.choice(n, size=2, replace=False)

        return sorted_pop[indices[0]], sorted_pop[indices[1]]

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

    def update_adaptive_parameters(self, population: list[Idea]) -> None:
        """Update adaptive parameters based on population metrics.

        Mathematical basis for adaptation:
        - Mutation rate: μ(t+1) = μ(t) * exp(τ * N(0,1)) where τ = 1/sqrt(2n)
        - Crossover rate: Adjusted based on fitness improvement rate
        - Temperature annealing: T(t+1) = T(t) * α, where α = 0.95
        """
        if not population:
            return

        # Update generation counter
        self.generation_count += 1

        # Calculate current metrics
        fitnesses = [idea.fitness for idea in population]
        np.mean(fitnesses)
        best_fitness = max(fitnesses)

        # Update fitness history
        self.fitness_history.append(best_fitness)

        # Calculate diversity (simple metric based on unique content prefixes)
        unique_prefixes = len(set(idea.content[:50] for idea in population))
        diversity = unique_prefixes / len(population)
        self.diversity_history.append(diversity)

        # Check for stagnation
        if len(self.fitness_history) > 3:
            recent_improvement = self.fitness_history[-1] - self.fitness_history[-4]
            if recent_improvement < 0.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        # Adaptive mutation rate
        if diversity < 0.3:
            # Low diversity - increase mutation
            self.parameters.mutation_rate = min(0.5, self.parameters.mutation_rate * 1.2)
        elif diversity > 0.7 and self.stagnation_counter == 0:
            # High diversity and improving - decrease mutation
            self.parameters.mutation_rate = max(0.05, self.parameters.mutation_rate * 0.9)

        # Adaptive crossover rate
        improvement_rate = 0.0
        if self.stagnation_counter > 2:
            # Stagnating - increase crossover
            self.parameters.crossover_rate = min(0.95, self.parameters.crossover_rate * 1.1)
        elif len(self.fitness_history) > 1:
            improvement_rate = (self.fitness_history[-1] - self.fitness_history[-2]) / max(0.001, self.fitness_history[-2])
            if improvement_rate > 0.1:
                # Good improvement - maintain current rate
                pass
            else:
                # Poor improvement - slightly increase
                self.parameters.crossover_rate = min(0.9, self.parameters.crossover_rate * 1.05)

        # Temperature annealing for Boltzmann selection
        self.boltzmann_temperature = max(0.1, self.boltzmann_temperature * 0.95)

        # Adaptive elitism
        if self.stagnation_counter > 3:
            # Reduce elitism when stagnating
            self.parameters.elitism_count = max(1, self.parameters.elitism_count - 1)
        elif improvement_rate > 0.15 and self.parameters.elitism_count < self.parameters.population_size // 4:
            # Increase elitism when improving rapidly
            self.parameters.elitism_count += 1

    def calculate_fitness_sharing(self, population: list[Idea], sigma_share: float = 0.1) -> dict[str, float]:
        """Calculate shared fitness values using fitness sharing.

        Mathematical basis: f'(i) = f(i) / Σ(sh(d_ij))
        where sh(d) = 1 - (d/σ)^α if d < σ, else 0
        """
        shared_fitness = {}
        n = len(population)

        if n == 0:
            return shared_fitness

        # Calculate pairwise distances (using simple content similarity)
        # In production, this would use embeddings
        for i, idea_i in enumerate(population):
            niche_count = 0

            for j, idea_j in enumerate(population):
                if i != j:
                    # Simple distance metric based on content overlap
                    # Replace with embedding distance in production
                    distance = self._calculate_content_distance(idea_i.content, idea_j.content)

                    if distance < sigma_share:
                        # Sharing function with α = 1 (linear)
                        sharing = 1 - (distance / sigma_share)
                        niche_count += sharing

            # Add self (distance = 0)
            niche_count += 1

            # Calculate shared fitness
            shared_fitness[idea_i.id] = idea_i.fitness / niche_count

        return shared_fitness

    def _calculate_content_distance(self, content1: str, content2: str) -> float:
        """Simple content distance metric (0 to 1)."""
        # Tokenize and calculate Jaccard distance
        tokens1 = set(content1.lower().split())
        tokens2 = set(content2.lower().split())

        if not tokens1 and not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 1.0

        jaccard_similarity = intersection / union
        return 1 - jaccard_similarity

    def calculate_crowding_distance(self, population: list[Idea]) -> dict[str, float]:
        """Calculate crowding distance for each individual (NSGA-II style).

        Mathematical basis: Measures the perimeter of the cuboid formed by nearest neighbors
        in objective space. Preserves diversity by favoring isolated solutions.
        """
        n = len(population)
        if n <= 2:
            return {idea.id: float('inf') for idea in population}

        crowding_distances = {idea.id: 0.0 for idea in population}

        # Get objectives (using fitness components)
        objectives = ['relevance', 'novelty', 'feasibility']

        for obj in objectives:
            # Sort population by this objective
            sorted_pop = sorted(population, key=lambda x: x.scores.get(obj, 0))

            # Boundary points get infinite distance
            crowding_distances[sorted_pop[0].id] = float('inf')
            crowding_distances[sorted_pop[-1].id] = float('inf')

            # Calculate distance for intermediate points
            obj_range = sorted_pop[-1].scores.get(obj, 0) - sorted_pop[0].scores.get(obj, 0)

            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (sorted_pop[i + 1].scores.get(obj, 0) -
                               sorted_pop[i - 1].scores.get(obj, 0)) / obj_range
                    crowding_distances[sorted_pop[i].id] += distance

        return crowding_distances

    def create_next_generation(self, population: list[Idea],
                             probabilities: list[float],
                             generation: int,
                             selection_method: str = "tournament",
                             use_fitness_sharing: bool = False,
                             use_crowding: bool = False) -> list[Idea]:
        """Create next generation using genetic operations with optional diversity preservation."""
        # Update adaptive parameters
        self.update_adaptive_parameters(population)

        new_population: list[Idea] = []

        # Apply fitness sharing if enabled
        if use_fitness_sharing:
            shared_fitness = self.calculate_fitness_sharing(population)
            # Update probabilities based on shared fitness
            total_shared = sum(shared_fitness.values())
            if total_shared > 0:
                probabilities = [shared_fitness.get(idea.id, 0) / total_shared for idea in population]

        # Calculate crowding distances if using crowding
        crowding_distances = None
        if use_crowding:
            crowding_distances = self.calculate_crowding_distance(population)

        # Elitism: Keep top performers
        if self.parameters.elitism_count > 0:
            if use_crowding and crowding_distances:
                # Sort by fitness but prefer higher crowding distance for ties
                sorted_pop = sorted(population,
                                  key=lambda x: (x.fitness, crowding_distances.get(x.id, 0)),
                                  reverse=True)
            else:
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
            # Select parents using specified method
            parent1, parent2 = self.select_parents(population, probabilities, method=selection_method)

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

    def check_convergence(self, population: list[Idea], window_size: int = 5) -> bool:
        """Check if the population has converged using multiple criteria.

        Convergence criteria:
        1. Fitness plateau: Best fitness hasn't improved significantly
        2. Low diversity: Population has become too homogeneous
        3. Small fitness variance: All individuals have similar fitness
        """
        if len(self.fitness_history) < window_size:
            return False

        # Criterion 1: Fitness plateau
        recent_fitnesses = self.fitness_history[-window_size:]
        fitness_improvement = max(recent_fitnesses) - min(recent_fitnesses)
        fitness_converged = fitness_improvement < 0.01 * max(recent_fitnesses)

        # Criterion 2: Low diversity
        if self.diversity_history:
            recent_diversity = np.mean(self.diversity_history[-window_size:])
            diversity_converged = recent_diversity < 0.2
        else:
            diversity_converged = False

        # Criterion 3: Small fitness variance
        if population:
            fitnesses = [idea.fitness for idea in population]
            fitness_variance = np.var(fitnesses)
            mean_fitness = np.mean(fitnesses)
            normalized_variance = fitness_variance / (mean_fitness + 1e-6)
            variance_converged = normalized_variance < 0.01
        else:
            variance_converged = False

        # Converged if at least 2 criteria are met
        criteria_met = sum([fitness_converged, diversity_converged, variance_converged])

        if criteria_met >= 2:
            logger.info(f"Convergence detected: fitness_plateau={fitness_converged}, "
                       f"low_diversity={diversity_converged}, "
                       f"low_variance={variance_converged}")

        return criteria_met >= 2

    def get_selection_method_for_generation(self, generation: int) -> str:
        """Adaptively choose selection method based on evolution stage.

        Strategy:
        - Early generations: More exploratory (SUS, rank-based)
        - Middle generations: Balanced (tournament)
        - Late generations: More exploitative (Boltzmann with low temperature)
        - Stagnation: Switch strategies to escape local optima
        """
        total_generations = self.parameters.generations
        progress = generation / total_generations if total_generations > 0 else 0

        # Check for stagnation
        if self.stagnation_counter > 3:
            # Rotate through methods to escape stagnation
            methods = ["sus", "rank", "boltzmann", "tournament"]
            return methods[self.stagnation_counter % len(methods)]

        # Normal progression
        if progress < 0.3:
            # Early stage - exploration
            return "sus" if generation % 2 == 0 else "rank"
        elif progress < 0.7:
            # Middle stage - balanced
            return "tournament"
        else:
            # Late stage - exploitation
            return "boltzmann"
