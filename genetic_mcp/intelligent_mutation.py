"""Intelligent mutation strategies for genetic algorithms with learning and adaptation."""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .llm_client import LLMClient
from .models import Idea

logger = logging.getLogger(__name__)


class MutationStrategy(str, Enum):
    """Available mutation strategies."""
    RANDOM = "random"
    GUIDED = "guided"
    ADAPTIVE = "adaptive"
    MEMETIC = "memetic"
    CONTEXT_AWARE = "context_aware"
    DIRECTIONAL = "directional"
    COMPONENT_BASED = "component_based"
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_BASED = "gradient_based"
    IMPLEMENTATION_DEEPENING = "implementation_deepening"


@dataclass
class MutationMetrics:
    """Tracks performance metrics for mutations."""
    strategy: str
    fitness_before: float
    fitness_after: float
    generation: int
    component_modified: str
    success: bool = False
    improvement: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.improvement = self.fitness_after - self.fitness_before
        self.success = self.improvement > 0.01  # Threshold for meaningful improvement


@dataclass
class ComponentMutationRate:
    """Tracks mutation rates for different idea components."""
    component: str
    base_rate: float = 0.1
    current_rate: float = 0.1
    success_count: int = 0
    failure_count: int = 0
    adaptations: int = 0

    def update_rate(self, success: bool, adaptation_factor: float = 0.1):
        """Update mutation rate based on success/failure."""
        if success:
            self.success_count += 1
            # Increase rate slightly on success
            self.current_rate = min(0.8, self.current_rate * (1 + adaptation_factor))
        else:
            self.failure_count += 1
            # Decrease rate on failure
            self.current_rate = max(0.01, self.current_rate * (1 - adaptation_factor))

        self.adaptations += 1

    def get_success_rate(self) -> float:
        """Get success rate for this component."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class FitnessLandscape:
    """Represents the fitness landscape around an idea."""
    center_idea: Idea
    fitness_gradient: dict[str, float] = field(default_factory=dict)
    neighboring_points: list[tuple[str, float]] = field(default_factory=list)
    local_optima: bool = False
    landscape_roughness: float = 0.0
    promising_directions: list[str] = field(default_factory=list)

    def analyze_landscape(self, neighbor_ideas: list[Idea]):
        """Analyze the fitness landscape around the center idea."""
        if not neighbor_ideas:
            return

        center_fitness = self.center_idea.fitness
        fitness_values = []

        for neighbor in neighbor_ideas:
            fitness_diff = neighbor.fitness - center_fitness
            self.neighboring_points.append((neighbor.content, neighbor.fitness))
            fitness_values.append(neighbor.fitness)

            # Simple gradient estimation
            if fitness_diff > 0.05:  # Significant improvement
                self.promising_directions.append(neighbor.content)

        # Calculate landscape roughness (variance of fitness values)
        if fitness_values:
            self.landscape_roughness = np.var(fitness_values)

            # Detect local optima (all neighbors have lower fitness)
            max_neighbor_fitness = max(fitness_values)
            self.local_optima = center_fitness > max_neighbor_fitness


class IntelligentMutationManager:
    """Advanced mutation manager with learning and adaptation capabilities."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

        # Mutation history and learning
        self.mutation_history: list[MutationMetrics] = []
        self.successful_patterns: dict[str, list[str]] = {}
        self.component_rates: dict[str, ComponentMutationRate] = {}

        # Fitness landscape analysis
        self.landscape_cache: dict[str, FitnessLandscape] = {}

        # Strategy performance tracking
        self.strategy_performance: dict[str, dict[str, float]] = {
            strategy.value: {
                'success_rate': 0.0,
                'avg_improvement': 0.0,
                'usage_count': 0,
                'last_update': time.time()
            }
            for strategy in MutationStrategy
        }

        # Adaptive parameters
        self.temperature = 1.0  # For simulated annealing
        self.learning_rate = 0.1
        self.exploration_rate = 0.2

        # Component identification patterns
        self.component_patterns = {
            'problem_statement': ['problem', 'issue', 'challenge', 'need'],
            'solution_approach': ['solution', 'approach', 'method', 'technique'],
            'implementation': ['implement', 'build', 'create', 'develop'],
            'implementation_code': ['code', 'function', 'class', 'api', 'endpoint', 'algorithm'],
            'technical_stack': ['technology', 'framework', 'library', 'stack', 'tool', 'platform'],
            'architecture': ['architecture', 'design', 'structure', 'pattern', 'component', 'module'],
            'testing': ['test', 'testing', 'validation', 'verification', 'quality'],
            'deployment': ['deploy', 'deployment', 'production', 'infrastructure', 'hosting'],
            'benefits': ['benefit', 'advantage', 'improve', 'enhance'],
            'constraints': ['constraint', 'limitation', 'requirement', 'must'],
            'examples': ['example', 'instance', 'case', 'sample']
        }

    async def mutate(
        self,
        idea: Idea,
        all_ideas: list[Idea],
        generation: int,
        strategy: MutationStrategy | None = None,
        target_embedding: list[float] | None = None,
        detail_config: Any | None = None
    ) -> str:
        """Apply intelligent mutation to an idea."""
        start_time = time.time()
        original_content = idea.content
        original_fitness = idea.fitness

        # Select mutation strategy if not specified
        if strategy is None:
            strategy = self._select_optimal_strategy(idea, generation, detail_config)

        logger.debug(f"Applying {strategy} mutation to idea {idea.id}")

        # Apply the selected mutation strategy
        try:
            if strategy == MutationStrategy.GUIDED:
                mutated_content = await self._guided_mutation(idea, all_ideas, target_embedding)
            elif strategy == MutationStrategy.ADAPTIVE:
                mutated_content = await self._adaptive_mutation(idea, generation)
            elif strategy == MutationStrategy.MEMETIC:
                mutated_content = await self._memetic_mutation(idea, all_ideas)
            elif strategy == MutationStrategy.CONTEXT_AWARE:
                mutated_content = await self._context_aware_mutation(idea, all_ideas)
            elif strategy == MutationStrategy.DIRECTIONAL:
                mutated_content = await self._directional_mutation(idea, target_embedding)
            elif strategy == MutationStrategy.COMPONENT_BASED:
                mutated_content = await self._component_based_mutation(idea)
            elif strategy == MutationStrategy.HILL_CLIMBING:
                mutated_content = await self._hill_climbing_mutation(idea, all_ideas)
            elif strategy == MutationStrategy.SIMULATED_ANNEALING:
                mutated_content = await self._simulated_annealing_mutation(idea, generation)
            elif strategy == MutationStrategy.GRADIENT_BASED:
                mutated_content = await self._gradient_based_mutation(idea, all_ideas)
            elif strategy == MutationStrategy.IMPLEMENTATION_DEEPENING:
                mutated_content = await self._implementation_deepening_mutation(idea)
            else:  # RANDOM
                mutated_content = await self._random_mutation(idea)

            # Record mutation metrics (fitness will be updated later by evaluator)
            component_modified = self._identify_modified_component(original_content, mutated_content)

            mutation_metric = MutationMetrics(
                strategy=strategy.value,
                fitness_before=original_fitness,
                fitness_after=original_fitness,  # Will be updated later
                generation=generation,
                component_modified=component_modified
            )

            # Store idea_id for later fitness updates
            mutation_metric.idea_id = idea.id  # Add this attribute dynamically
            self.mutation_history.append(mutation_metric)

            logger.debug(f"Mutation completed in {time.time() - start_time:.3f}s")
            return mutated_content

        except Exception as e:
            logger.error(f"Mutation failed with strategy {strategy}: {e}")
            # Fallback to random mutation
            return await self._random_mutation(idea)

    def update_mutation_feedback(self, idea_id: str, new_fitness: float):
        """Update mutation performance based on fitness feedback."""
        # Find the most recent mutation for this idea
        for metric in reversed(self.mutation_history):
            if hasattr(metric, 'idea_id') and metric.idea_id == idea_id:
                metric.fitness_after = new_fitness
                metric.improvement = new_fitness - metric.fitness_before
                metric.success = metric.improvement > 0.01

                # Update strategy performance
                self._update_strategy_performance(metric)

                # Update component mutation rates
                self._update_component_rates(metric)

                # Learn from successful patterns
                if metric.success:
                    self._learn_successful_pattern(metric)

                break

    def _select_optimal_strategy(self, idea: Idea, generation: int, detail_config: Any | None = None) -> MutationStrategy:
        """Select the optimal mutation strategy based on learned performance."""
        # Check if detail_config favors implementation deepening
        if detail_config is not None:
            # Check if detail_config has high implementation requirements
            should_deepen = False

            # Check for explicit attributes
            if (hasattr(detail_config, 'require_code_examples') and detail_config.require_code_examples) or \
               (hasattr(detail_config, 'require_technical_specs') and detail_config.require_technical_specs):
                should_deepen = True
            elif hasattr(detail_config, 'level') and detail_config.level == 'high' and random.random() < 0.6:
                # For high detail level, favor implementation deepening with 60% probability
                should_deepen = True

            # If detail requirements are high, strongly favor implementation_deepening
            if should_deepen:
                # 70% chance to use implementation_deepening, 30% for other strategies
                if random.random() < 0.7:
                    return MutationStrategy.IMPLEMENTATION_DEEPENING
                # Otherwise use memetic or component-based which also help with details
                return random.choice([MutationStrategy.MEMETIC, MutationStrategy.COMPONENT_BASED])

        # Early generations: more exploration (but include implementation_deepening)
        if generation < 3:
            exploration_strategies = [
                MutationStrategy.RANDOM,
                MutationStrategy.CONTEXT_AWARE,
                MutationStrategy.COMPONENT_BASED,
                MutationStrategy.IMPLEMENTATION_DEEPENING  # Add to early exploration
            ]
            return random.choice(exploration_strategies)

        # Check if we're in a local optimum
        landscape = self._analyze_fitness_landscape(idea)
        if landscape and landscape.local_optima:
            # Use more disruptive strategies
            disruptive_strategies = [
                MutationStrategy.SIMULATED_ANNEALING,
                MutationStrategy.MEMETIC,
                MutationStrategy.DIRECTIONAL,
                MutationStrategy.IMPLEMENTATION_DEEPENING  # Can help escape local optima by adding detail
            ]
            return random.choice(disruptive_strategies)

        # Use UCB1 algorithm for strategy selection
        if random.random() < self.exploration_rate:
            # Exploration: try strategies with fewer attempts
            min_usage = min(perf['usage_count'] for perf in self.strategy_performance.values())
            candidates = [
                MutationStrategy(strategy) for strategy, perf in self.strategy_performance.items()
                if perf['usage_count'] == min_usage
            ]
            return random.choice(candidates)

        # Exploitation: choose best performing strategy
        best_strategy = max(
            self.strategy_performance.items(),
            key=lambda x: x[1]['success_rate'] + x[1]['avg_improvement']
        )[0]

        return MutationStrategy(best_strategy)

    async def _guided_mutation(
        self,
        idea: Idea,
        all_ideas: list[Idea],
        target_embedding: list[float] | None
    ) -> str:
        """Mutation guided by fitness landscape analysis."""
        landscape = self._analyze_fitness_landscape(idea, all_ideas)

        if landscape and landscape.promising_directions:
            # Use promising directions from landscape analysis
            direction = random.choice(landscape.promising_directions)

            if self.llm_client:
                prompt = f"""
                You are helping evolve ideas using fitness landscape guidance.

                Original idea: {idea.content}

                Promising direction observed: {direction}

                Modify the original idea to incorporate elements that led to the promising direction,
                while maintaining the core concept. Focus on incremental improvements and especially
                on adding implementation details such as:
                - Specific technology names and frameworks
                - Code examples or pseudocode
                - Step-by-step implementation approaches
                - Data structures and API designs
                - Testing and deployment considerations

                Return only the modified idea with enhanced implementation details:
                """

                try:
                    return await self.llm_client.generate(
                        prompt,
                        temperature=0.6,
                        max_tokens=1500
                    )
                except Exception as e:
                    logger.error(f"Guided mutation LLM call failed: {e}")

        # Fallback to component-based mutation
        return await self._component_based_mutation(idea)

    async def _adaptive_mutation(self, idea: Idea, generation: int) -> str:
        """Mutation with adaptive rates based on component success."""
        components = self._extract_components(idea.content)

        # Select component to mutate based on adaptive rates
        component_to_mutate = self._select_component_to_mutate(components)

        if not component_to_mutate or not self.llm_client:
            return await self._random_mutation(idea)

        mutation_strength = self._calculate_mutation_strength(component_to_mutate, generation)

        prompt = f"""
        You are performing adaptive mutation on a specific component of an idea.

        Original idea: {idea.content}

        Component to modify: {component_to_mutate}
        Mutation strength: {mutation_strength} (0=minimal, 1=major changes)

        Modify only the identified component with the specified strength while keeping
        the rest of the idea unchanged. Return the complete modified idea:
        """

        try:
            temperature = 0.5 + (mutation_strength * 0.3)  # Higher temp for stronger mutations
            return await self.llm_client.generate(
                prompt,
                temperature=temperature,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Adaptive mutation failed: {e}")
            return await self._random_mutation(idea)

    async def _memetic_mutation(self, idea: Idea, all_ideas: list[Idea]) -> str:
        """Memetic algorithm with local search optimization."""
        # First apply standard mutation
        mutated = await self._component_based_mutation(idea)

        if not self.llm_client:
            return mutated

        # Then apply local search optimization
        prompt = f"""
        You are performing local search optimization on a mutated idea.

        Original idea: {idea.content}
        Mutated idea: {mutated}

        Apply local search to find small improvements to the mutated idea.
        Look for and prioritize:
        1. **MISSING IMPLEMENTATION DETAILS** - Add specific code examples, technology names, API specifications
        2. **MISSING IMPLEMENTATION DETAILS** - Include numbered implementation steps and concrete technical approaches
        3. **MISSING IMPLEMENTATION DETAILS** - Specify data models, testing strategies, and deployment considerations
        4. More precise terminology (replace generic terms with specific technology names)
        5. Better structure or organization
        6. Clearer explanations with concrete examples

        IMPORTANT: Focus heavily on adding implementation details. If the idea lacks code examples,
        specific technologies, or concrete steps, add them. Replace vague terms like "using modern
        frameworks" with specific choices like "using React and FastAPI".

        Return the locally optimized version with enhanced implementation details:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.4,  # Lower temperature for refinement
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Memetic mutation failed: {e}")
            return mutated

    async def _context_aware_mutation(self, idea: Idea, all_ideas: list[Idea]) -> str:
        """Mutation that considers the context of other ideas in the population."""
        # Analyze population diversity and gaps
        population_themes = self._analyze_population_themes(all_ideas)
        underrepresented_themes = self._identify_gaps(population_themes)

        if not underrepresented_themes or not self.llm_client:
            return await self._component_based_mutation(idea)

        theme_to_incorporate = random.choice(underrepresented_themes)

        prompt = f"""
        You are performing context-aware mutation based on population analysis.

        Original idea: {idea.content}

        Population analysis shows these themes are underrepresented:
        {', '.join(underrepresented_themes)}

        Modify the idea to incorporate elements of: {theme_to_incorporate}

        Maintain the core concept while adding the missing perspective.
        Return the modified idea:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.7,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Context-aware mutation failed: {e}")
            return await self._component_based_mutation(idea)

    async def _directional_mutation(
        self,
        idea: Idea,
        target_embedding: list[float] | None
    ) -> str:
        """Mutation that moves toward promising regions of the search space."""
        if not self.llm_client:
            return await self._random_mutation(idea)

        # Find the most successful recent mutations
        recent_successes = [
            m for m in self.mutation_history[-20:]  # Last 20 mutations
            if m.success and m.improvement > 0.1
        ]

        if not recent_successes:
            return await self._guided_mutation(idea, [], target_embedding)

        # Use the most successful mutation pattern
        best_mutation = max(recent_successes, key=lambda x: x.improvement)

        prompt = f"""
        You are performing directional mutation toward a promising search region.

        Original idea: {idea.content}

        A recent successful mutation (improvement: {best_mutation.improvement:.3f})
        modified the "{best_mutation.component_modified}" component using
        the {best_mutation.strategy} strategy.

        Apply a similar type of modification to the same or related component
        in this idea. Look for analogous improvements.

        Return the directionally modified idea:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.6,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Directional mutation failed: {e}")
            return await self._component_based_mutation(idea)

    async def _component_based_mutation(self, idea: Idea) -> str:
        """Mutation targeting specific components with learned rates."""
        components = self._extract_components(idea.content)

        if not components:
            return await self._random_mutation(idea)

        # Select component based on mutation rates
        component_to_mutate = self._select_component_to_mutate(components)

        if not self.llm_client or not component_to_mutate:
            return self._fallback_component_mutation(idea.content, components)

        # Get component-specific mutation rate
        mutation_rate = self.component_rates.get(
            component_to_mutate,
            ComponentMutationRate(component_to_mutate)
        ).current_rate

        # Check if this is an implementation-related component
        implementation_components = {
            'implementation', 'implementation_code', 'technical_stack',
            'architecture', 'testing', 'deployment'
        }
        is_implementation = component_to_mutate in implementation_components

        implementation_guidance = ""
        if is_implementation:
            implementation_guidance = """

            FOCUS ON IMPLEMENTATION DETAILS:
            - Add specific code examples or pseudocode
            - Replace generic technology references with specific names (e.g., "React" instead of "modern framework")
            - Include numbered implementation steps
            - Specify APIs, data models, and interfaces
            - Add testing approaches and deployment considerations
            """

        prompt = f"""
        You are performing component-based mutation on a specific part of an idea.

        Original idea: {idea.content}

        Target component: {component_to_mutate}
        Mutation intensity: {mutation_rate}{implementation_guidance}

        Modify only the {component_to_mutate} component with the specified intensity.
        Keep other parts of the idea unchanged.

        Return the complete modified idea:
        """

        try:
            temperature = 0.5 + mutation_rate * 0.3
            return await self.llm_client.generate(
                prompt,
                temperature=temperature,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Component-based mutation failed: {e}")
            return self._fallback_component_mutation(idea.content, components)

    async def _hill_climbing_mutation(self, idea: Idea, all_ideas: list[Idea]) -> str:
        """Hill climbing mutation for local optimization."""
        if not self.llm_client:
            return await self._random_mutation(idea)

        # Generate multiple small mutations and pick the best direction
        candidates = []

        for _ in range(3):  # Try 3 small mutations
            prompt = f"""
            You are performing hill climbing mutation - make a small improvement.

            Original idea: {idea.content}

            Make ONE small, specific improvement to this idea. Focus on:
            - Adding a missing detail
            - Clarifying an ambiguous point
            - Improving structure
            - Enhancing feasibility

            Return only the slightly improved version:
            """

            try:
                candidate = await self.llm_client.generate(
                    prompt,
                    temperature=0.3,  # Low temperature for small changes
                    max_tokens=1500
                )
                candidates.append(candidate)
            except Exception as e:
                logger.error(f"Hill climbing candidate generation failed: {e}")

        # Return the first successful candidate (in practice, would evaluate all)
        return candidates[0] if candidates else await self._random_mutation(idea)

    async def _simulated_annealing_mutation(self, idea: Idea, generation: int) -> str:
        """Simulated annealing mutation with temperature cooling."""
        # Update temperature based on generation
        self.temperature = max(0.1, self.temperature * 0.95)

        if not self.llm_client:
            return await self._random_mutation(idea)

        # Higher temperature = more radical mutations

        prompt = f"""
        You are performing simulated annealing mutation.

        Original idea: {idea.content}

        Temperature: {self.temperature:.3f} (0=conservative, 1=radical)

        Based on the temperature, modify the idea:
        - Low temperature (< 0.3): Make small refinements
        - Medium temperature (0.3-0.7): Make moderate changes
        - High temperature (> 0.7): Make significant transformations

        Return the temperature-appropriate mutation:
        """

        try:
            # Temperature affects LLM temperature too
            llm_temp = 0.4 + (self.temperature * 0.5)
            return await self.llm_client.generate(
                prompt,
                temperature=llm_temp,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Simulated annealing mutation failed: {e}")
            return await self._random_mutation(idea)

    async def _gradient_based_mutation(self, idea: Idea, all_ideas: list[Idea]) -> str:
        """Mutation guided by fitness gradient estimation."""
        # Estimate fitness gradient from nearby ideas
        gradient_info = self._estimate_fitness_gradient(idea, all_ideas)

        if not gradient_info or not self.llm_client:
            return await self._component_based_mutation(idea)

        best_direction, improvement = gradient_info

        prompt = f"""
        You are performing gradient-based mutation guided by fitness landscape.

        Original idea: {idea.content}

        Fitness gradient analysis suggests moving toward this direction:
        {best_direction}

        Expected improvement potential: {improvement:.3f}

        Modify the idea to move in this fitness-improving direction while
        maintaining its core concept.

        Return the gradient-guided mutation:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.6,
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Gradient-based mutation failed: {e}")
            return await self._component_based_mutation(idea)

    async def _implementation_deepening_mutation(self, idea: Idea) -> str:
        """Mutation that specifically adds implementation details to ideas."""
        if not self.llm_client:
            return await self._random_mutation(idea)

        # Analyze what implementation details are missing
        content_lower = idea.content.lower()
        missing_details = []

        if not any(keyword in content_lower for keyword in ['code', 'function', 'class', 'def ', 'const ', 'let ']):
            missing_details.append("code examples or pseudocode")

        if not any(keyword in content_lower for keyword in ['react', 'vue', 'django', 'flask', 'fastapi', 'express', 'spring', 'framework']):
            missing_details.append("specific technology/framework names")

        if not any(keyword in content_lower for keyword in ['step 1', 'first,', 'then,', 'finally,', '1.', '2.']):
            missing_details.append("numbered implementation steps")

        if not any(keyword in content_lower for keyword in ['api', 'endpoint', 'interface', 'schema', 'model']):
            missing_details.append("API/data model specifications")

        if not any(keyword in content_lower for keyword in ['test', 'unit test', 'integration test', 'validation']):
            missing_details.append("testing approach")

        if not any(keyword in content_lower for keyword in ['deploy', 'docker', 'kubernetes', 'aws', 'azure', 'cloud']):
            missing_details.append("deployment considerations")

        missing_summary = ", ".join(missing_details) if missing_details else "general implementation detail"

        prompt = f"""
        You are performing IMPLEMENTATION DEEPENING mutation to add concrete technical details.

        Original idea: {idea.content}

        Analysis shows this idea is missing: {missing_summary}

        Your task is to SIGNIFICANTLY ENHANCE this idea with implementation details:

        1. **CODE EXAMPLES**: Add specific code snippets or pseudocode showing key implementations
           - Use actual syntax (Python, JavaScript, etc.)
           - Show data structures, function signatures, or class definitions

        2. **SPECIFIC TECHNOLOGIES**: Replace generic terms with specific technology names
           - Instead of "database", specify "PostgreSQL" or "MongoDB"
           - Instead of "web framework", specify "FastAPI" or "React"
           - Instead of "cloud platform", specify "AWS" or "Google Cloud"

        3. **IMPLEMENTATION STEPS**: Provide a numbered, step-by-step implementation approach
           - Step 1: Specific action with technical details
           - Step 2: Next action with configuration details
           - Step 3: Integration steps with code references

        4. **API/DATA SPECIFICATIONS**: Define specific APIs, endpoints, or data models
           - Example: "POST /api/v1/ideas endpoint accepting JSON: {{title, description, tags[]}}"
           - Example: "User model: {{id: UUID, email: string, created_at: timestamp}}"

        5. **TESTING APPROACH**: Specify how to test the implementation
           - Unit tests for core logic using pytest or jest
           - Integration tests for API endpoints
           - Example test cases

        6. **DEPLOYMENT DETAILS**: Describe deployment setup
           - Docker containerization with Dockerfile
           - CI/CD pipeline (GitHub Actions, Jenkins)
           - Cloud hosting specifics (AWS ECS, Heroku, Vercel)

        CRITICAL: Do not just add vague statements. Add CONCRETE, SPECIFIC, TECHNICAL details.
        Transform abstract ideas into implementable solutions with clear technical choices.

        Return the deeply enhanced idea with rich implementation details:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.5,  # Moderate temperature for structured output
                max_tokens=2000  # More tokens for detailed implementations
            )
        except Exception as e:
            logger.error(f"Implementation deepening mutation failed: {e}")
            return await self._component_based_mutation(idea)

    async def _random_mutation(self, idea: Idea) -> str:
        """Fallback random mutation."""
        if not self.llm_client:
            return self._basic_random_mutation(idea.content)

        prompt = f"""
        You are performing random mutation on an idea.

        Original idea: {idea.content}

        Make a random but meaningful modification to this idea.
        The change should be significant enough to explore new areas
        but not completely destroy the original concept.

        Return the randomly mutated idea:
        """

        try:
            return await self.llm_client.generate(
                prompt,
                temperature=0.8,  # High temperature for randomness
                max_tokens=1500
            )
        except Exception as e:
            logger.error(f"Random mutation failed: {e}")
            return self._basic_random_mutation(idea.content)

    def _extract_components(self, content: str) -> dict[str, str]:
        """Extract semantic components from idea content."""
        components = {}
        content_lower = content.lower()

        # Use pattern matching to identify components
        for component_type, patterns in self.component_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    # Extract sentences containing the pattern
                    sentences = content.split('.')
                    matching_sentences = [
                        s.strip() for s in sentences
                        if pattern in s.lower() and s.strip()
                    ]
                    if matching_sentences:
                        components[component_type] = '. '.join(matching_sentences)
                        break

        # If no patterns match, create generic components
        if not components:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if sentences:
                mid = len(sentences) // 2
                components['first_half'] = '. '.join(sentences[:mid])
                components['second_half'] = '. '.join(sentences[mid:])

        return components

    def _select_component_to_mutate(self, components: dict[str, str]) -> str | None:
        """Select which component to mutate based on learned rates."""
        if not components:
            return None

        # If we have learned rates, use them for selection
        if any(comp in self.component_rates for comp in components):
            # Weighted selection based on success rates
            weights = []
            component_names = list(components.keys())

            for comp in component_names:
                if comp in self.component_rates:
                    rate = self.component_rates[comp]
                    # Higher rate for components with better success
                    weight = rate.current_rate * (1 + rate.get_success_rate())
                else:
                    weight = 0.1  # Default weight
                weights.append(weight)

            # Weighted random selection
            if sum(weights) > 0:
                probabilities = [w / sum(weights) for w in weights]
                return np.random.choice(component_names, p=probabilities)

        # Random selection if no learned rates
        return random.choice(list(components.keys()))

    def _calculate_mutation_strength(self, component: str, generation: int) -> float:
        """Calculate mutation strength based on component performance and generation."""
        base_strength = 0.5

        # Adjust based on generation (more conservative in later generations)
        generation_factor = max(0.1, 1.0 - (generation * 0.05))

        # Adjust based on component success rate
        if component in self.component_rates:
            success_rate = self.component_rates[component].get_success_rate()
            # Higher success rate = more aggressive mutations
            success_factor = 0.5 + success_rate
        else:
            success_factor = 1.0

        return min(1.0, base_strength * generation_factor * success_factor)

    def _analyze_fitness_landscape(
        self,
        idea: Idea,
        all_ideas: list[Idea] | None = None
    ) -> FitnessLandscape | None:
        """Analyze fitness landscape around an idea."""
        if idea.id in self.landscape_cache:
            return self.landscape_cache[idea.id]

        if not all_ideas:
            return None

        # Find neighboring ideas (simple content similarity)
        neighbors = []
        idea_words = set(idea.content.lower().split())

        for other_idea in all_ideas:
            if other_idea.id == idea.id:
                continue

            other_words = set(other_idea.content.lower().split())
            overlap = len(idea_words & other_words) / len(idea_words | other_words)

            if overlap > 0.3:  # Similar enough to be neighbors
                neighbors.append(other_idea)

        landscape = FitnessLandscape(center_idea=idea)
        landscape.analyze_landscape(neighbors)

        # Cache the result
        self.landscape_cache[idea.id] = landscape

        return landscape

    def _analyze_population_themes(self, all_ideas: list[Idea]) -> dict[str, int]:
        """Analyze common themes in the population."""
        themes = {}

        for idea in all_ideas:
            words = idea.content.lower().split()
            # Simple keyword extraction
            for word in words:
                if len(word) > 3:  # Skip short words
                    themes[word] = themes.get(word, 0) + 1

        # Return most common themes
        return dict(sorted(themes.items(), key=lambda x: x[1], reverse=True)[:20])

    def _identify_gaps(self, themes: dict[str, int]) -> list[str]:
        """Identify underrepresented themes."""
        if not themes:
            return []

        avg_count = sum(themes.values()) / len(themes)
        underrepresented = [
            theme for theme, count in themes.items()
            if count < avg_count * 0.5
        ]

        return underrepresented[:5]  # Top 5 gaps

    def _estimate_fitness_gradient(
        self,
        idea: Idea,
        all_ideas: list[Idea]
    ) -> tuple[str, float] | None:
        """Estimate fitness gradient from nearby ideas."""
        if len(all_ideas) < 3:
            return None

        # Find ideas with higher fitness
        better_ideas = [
            other for other in all_ideas
            if other.fitness > idea.fitness and other.id != idea.id
        ]

        if not better_ideas:
            return None

        # Find the best nearby idea
        best_idea = max(better_ideas, key=lambda x: x.fitness)
        improvement = best_idea.fitness - idea.fitness

        return best_idea.content, improvement

    def _identify_modified_component(self, original: str, modified: str) -> str:
        """Identify which component was modified."""
        # Simple heuristic: find which sentences changed
        orig_sentences = set(original.split('.'))
        mod_sentences = set(modified.split('.'))

        if orig_sentences != mod_sentences:
            # Check component patterns
            for component_type, patterns in self.component_patterns.items():
                for pattern in patterns:
                    if pattern in modified.lower() and pattern not in original.lower() or pattern in original.lower() and pattern not in modified.lower():
                        return component_type

        return 'unknown'

    def _update_strategy_performance(self, metric: MutationMetrics):
        """Update strategy performance metrics."""
        strategy = metric.strategy
        perf = self.strategy_performance[strategy]

        perf['usage_count'] += 1

        # Update success rate with exponential moving average
        alpha = 0.1
        new_success = 1.0 if metric.success else 0.0
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * new_success

        # Update average improvement
        perf['avg_improvement'] = (1 - alpha) * perf['avg_improvement'] + alpha * metric.improvement
        perf['last_update'] = time.time()

    def _update_component_rates(self, metric: MutationMetrics):
        """Update component mutation rates based on feedback."""
        component = metric.component_modified

        if component not in self.component_rates:
            self.component_rates[component] = ComponentMutationRate(component)

        self.component_rates[component].update_rate(metric.success)

    def _learn_successful_pattern(self, metric: MutationMetrics):
        """Learn from successful mutation patterns."""
        if metric.improvement > 0.1:  # Significant improvement
            strategy = metric.strategy
            component = metric.component_modified

            if strategy not in self.successful_patterns:
                self.successful_patterns[strategy] = []

            pattern = f"{component}:{metric.improvement:.3f}"
            self.successful_patterns[strategy].append(pattern)

            # Keep only recent successful patterns
            self.successful_patterns[strategy] = self.successful_patterns[strategy][-10:]

    def _fallback_component_mutation(self, content: str, components: dict[str, str]) -> str:
        """Simple fallback mutation when LLM is not available."""
        if not components:
            return self._basic_random_mutation(content)

        # Select random component and apply simple transformation
        component_name = random.choice(list(components.keys()))
        component_text = components[component_name]

        # Simple transformations
        transformations = [
            lambda x: x.replace("could", "should"),
            lambda x: x.replace("might", "will"),
            lambda x: x.replace("basic", "advanced"),
            lambda x: x + " This approach offers significant benefits.",
            lambda x: "Enhanced " + x.lower(),
        ]

        transform = random.choice(transformations)
        modified_component = transform(component_text)

        return content.replace(component_text, modified_component)

    def _basic_random_mutation(self, content: str) -> str:
        """Very basic random mutation fallback."""
        sentences = content.split('.')
        if len(sentences) > 2:
            # Shuffle middle sentences
            middle = sentences[1:-1]
            random.shuffle(middle)
            return sentences[0] + '. ' + '. '.join(middle) + '. ' + sentences[-1]
        return content

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'strategy_performance': self.strategy_performance,
            'component_rates': {
                comp: {
                    'current_rate': rate.current_rate,
                    'success_rate': rate.get_success_rate(),
                    'adaptations': rate.adaptations
                }
                for comp, rate in self.component_rates.items()
            },
            'total_mutations': len(self.mutation_history),
            'successful_mutations': sum(1 for m in self.mutation_history if m.success),
            'avg_improvement': np.mean([m.improvement for m in self.mutation_history]) if self.mutation_history else 0,
            'successful_patterns': self.successful_patterns,
            'current_temperature': self.temperature
        }

    def reset_adaptation(self):
        """Reset adaptation parameters for new session."""
        self.temperature = 1.0
        self.mutation_history = []
        self.landscape_cache = {}

        # Reset strategy performance tracking
        for strategy in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'success_rate': 0.0,
                'avg_improvement': 0.0,
                'usage_count': 0,
                'last_update': time.time()
            }
