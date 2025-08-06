"""Advanced crossover operators for genetic algorithm-based idea generation."""

import logging
import random
import re
from enum import Enum
from typing import Any

import numpy as np

from .llm_client import LLMClient
from .models import Idea

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some advanced crossover features will use fallback methods.")


class CrossoverOperator(str, Enum):
    """Available crossover operators."""
    SEMANTIC = "semantic"           # LLM-guided semantic blending
    MULTI_POINT = "multi_point"     # Split ideas at multiple points
    UNIFORM = "uniform"             # Use learned masks for selection
    EDGE_RECOMBINATION = "edge_recombination"  # Preserve concept relationships
    ORDER_BASED = "order_based"     # Maintain logical flow
    BLEND = "blend"                 # Fitness-weighted combination
    ADAPTIVE = "adaptive"           # Choose operator dynamically
    CONCEPT_MAPPING = "concept_mapping"  # Map and recombine concepts
    SYNTACTIC = "syntactic"         # Grammar-aware crossover
    HIERARCHICAL = "hierarchical"   # Tree-based structure crossover


class ConceptNode:
    """Represents a concept in an idea's structure."""
    def __init__(self, content: str, importance: float = 1.0, position: int = 0):
        self.content = content
        self.importance = importance
        self.position = position
        self.connections: list[ConceptNode] = []
        self.semantic_embedding: list[float] = []

    def add_connection(self, node: 'ConceptNode', weight: float = 1.0):
        """Add a connection to another concept node."""
        if node not in self.connections:
            self.connections.append(node)
            node.connections.append(self)


class CrossoverMetrics:
    """Tracks performance metrics for crossover operators."""
    def __init__(self):
        self.operator_usage: dict[str, int] = {}
        self.operator_success: dict[str, int] = {}
        self.operator_fitness_improvements: dict[str, list[float]] = {}
        self.parent_similarity_scores: list[float] = []
        self.offspring_diversity_scores: list[float] = []

    def record_usage(self, operator: str, success: bool, fitness_improvement: float = 0.0):
        """Record usage statistics for an operator."""
        self.operator_usage[operator] = self.operator_usage.get(operator, 0) + 1
        if success:
            self.operator_success[operator] = self.operator_success.get(operator, 0) + 1
            if operator not in self.operator_fitness_improvements:
                self.operator_fitness_improvements[operator] = []
            self.operator_fitness_improvements[operator].append(fitness_improvement)

    def get_operator_performance(self, operator: str) -> dict[str, float]:
        """Get performance metrics for an operator."""
        usage = self.operator_usage.get(operator, 0)
        success = self.operator_success.get(operator, 0)

        performance = {
            "usage_count": usage,
            "success_rate": success / usage if usage > 0 else 0.0,
            "avg_fitness_improvement": 0.0
        }

        if operator in self.operator_fitness_improvements:
            improvements = self.operator_fitness_improvements[operator]
            if improvements:
                performance["avg_fitness_improvement"] = sum(improvements) / len(improvements)

        return performance


class AdvancedCrossoverManager:
    """Manages advanced crossover operations for genetic algorithms."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client
        self.metrics = CrossoverMetrics()

        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.tfidf_vectorizer = None

        self.concept_cache: dict[str, list[ConceptNode]] = {}

        # Adaptive selection parameters
        self.operator_weights: dict[str, float] = {op.value: 1.0 for op in CrossoverOperator}
        self.exploration_factor = 0.1
        self.min_usage_threshold = 5  # Minimum uses before performance-based selection

    async def crossover(
        self,
        parent1: Idea,
        parent2: Idea,
        operator: CrossoverOperator = CrossoverOperator.ADAPTIVE,
        generation: int = 0,
        **kwargs
    ) -> tuple[str, str]:
        """Execute crossover operation between two parents."""

        if operator == CrossoverOperator.ADAPTIVE:
            operator = self._select_adaptive_operator(parent1, parent2, generation)

        logger.debug(f"Performing {operator.value} crossover between {parent1.id} and {parent2.id}")

        try:
            # Execute the chosen crossover operator
            if operator == CrossoverOperator.SEMANTIC:
                offspring1, offspring2 = await self._semantic_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.MULTI_POINT:
                offspring1, offspring2 = await self._multi_point_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.UNIFORM:
                offspring1, offspring2 = await self._uniform_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.EDGE_RECOMBINATION:
                offspring1, offspring2 = await self._edge_recombination_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.ORDER_BASED:
                offspring1, offspring2 = await self._order_based_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.BLEND:
                offspring1, offspring2 = await self._blend_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.CONCEPT_MAPPING:
                offspring1, offspring2 = await self._concept_mapping_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.SYNTACTIC:
                offspring1, offspring2 = await self._syntactic_crossover(parent1, parent2, **kwargs)
            elif operator == CrossoverOperator.HIERARCHICAL:
                offspring1, offspring2 = await self._hierarchical_crossover(parent1, parent2, **kwargs)
            else:
                # Fallback to semantic crossover
                offspring1, offspring2 = await self._semantic_crossover(parent1, parent2, **kwargs)

            # Record usage (fitness improvement will be recorded later when fitness is calculated)
            self.metrics.record_usage(operator.value, True)

            return offspring1, offspring2, operator.value

        except Exception as e:
            logger.error(f"Crossover operation {operator.value} failed: {e}")
            self.metrics.record_usage(operator.value, False)

            # Fallback to simple crossover
            offspring1, offspring2 = self._simple_fallback_crossover(parent1, parent2)
            return offspring1, offspring2, "fallback"

    def _select_adaptive_operator(self, parent1: Idea, parent2: Idea, generation: int) -> CrossoverOperator:
        """Intelligently select crossover operator based on parent characteristics and performance."""

        # Analyze parent characteristics
        similarity = self._calculate_content_similarity(parent1.content, parent2.content)
        parent1_complexity = self._estimate_content_complexity(parent1.content)
        parent2_complexity = self._estimate_content_complexity(parent2.content)
        avg_complexity = (parent1_complexity + parent2_complexity) / 2

        # Early generation heuristics
        if generation <= 2:
            if similarity > 0.8:
                return CrossoverOperator.UNIFORM  # High similarity, use uniform
            elif avg_complexity > 0.7:
                return CrossoverOperator.CONCEPT_MAPPING  # Complex ideas, use concept mapping
            else:
                return CrossoverOperator.SEMANTIC  # Default to semantic

        # Performance-based selection for later generations
        viable_operators = self._get_viable_operators(similarity, avg_complexity)

        # Use Upper Confidence Bound (UCB) for exploration-exploitation balance
        if all(self.metrics.operator_usage.get(op.value, 0) >= self.min_usage_threshold
               for op in viable_operators):
            return self._ucb_operator_selection(viable_operators, generation)
        else:
            # Still exploring - choose less used operators
            return min(viable_operators, key=lambda op: self.metrics.operator_usage.get(op.value, 0))

    def _get_viable_operators(self, similarity: float, complexity: float) -> list[CrossoverOperator]:
        """Get viable operators based on content characteristics."""
        operators = []

        # Always available
        operators.extend([CrossoverOperator.SEMANTIC, CrossoverOperator.MULTI_POINT, CrossoverOperator.BLEND])

        # Add based on similarity
        if similarity < 0.3:
            operators.append(CrossoverOperator.CONCEPT_MAPPING)  # Low similarity, try concept mapping
        elif similarity > 0.7:
            operators.extend([CrossoverOperator.UNIFORM, CrossoverOperator.EDGE_RECOMBINATION])

        # Add based on complexity
        if complexity > 0.6:
            operators.extend([CrossoverOperator.HIERARCHICAL, CrossoverOperator.SYNTACTIC])

        if complexity < 0.4:
            operators.append(CrossoverOperator.ORDER_BASED)

        return operators

    def _ucb_operator_selection(self, operators: list[CrossoverOperator], generation: int) -> CrossoverOperator:
        """Select operator using Upper Confidence Bound algorithm."""
        ucb_scores = {}

        for operator in operators:
            performance = self.metrics.get_operator_performance(operator.value)
            avg_reward = performance["avg_fitness_improvement"]
            usage_count = performance["usage_count"]

            if usage_count == 0:
                ucb_scores[operator] = float('inf')  # Prioritize unexplored operators
            else:
                exploration_term = self.exploration_factor * np.sqrt(np.log(generation + 1) / usage_count)
                ucb_scores[operator] = avg_reward + exploration_term

        return max(ucb_scores.keys(), key=lambda op: ucb_scores[op])

    async def _semantic_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """LLM-guided semantic crossover with enhanced prompting."""
        if not self.llm_client:
            return self._simple_fallback_crossover(parent1, parent2)

        # Enhanced crossover prompt with specific instructions
        crossover_prompt = f"""
        You are performing genetic algorithm crossover to create two innovative offspring ideas.

        PARENT 1 (Fitness: {parent1.fitness:.3f}): {parent1.content}

        PARENT 2 (Fitness: {parent2.fitness:.3f}): {parent2.content}

        Create two distinct offspring by intelligently combining the best aspects of both parents:

        OFFSPRING_1: Should emphasize the strongest concepts from Parent 1 while incorporating complementary elements from Parent 2. Focus on preserving the core innovation of Parent 1.

        OFFSPRING_2: Should emphasize the strongest concepts from Parent 2 while incorporating complementary elements from Parent 1. Focus on preserving the core innovation of Parent 2.

        Guidelines:
        - Maintain coherence and logical flow in both offspring
        - Combine concepts creatively, don't just concatenate
        - Preserve the innovative essence of each parent
        - Ensure both offspring are distinct and viable
        - Keep similar length and complexity to parents

        Format your response exactly as:
        OFFSPRING_1: [detailed content]
        OFFSPRING_2: [detailed content]
        """

        try:
            temperature = random.uniform(0.6, 0.8)  # Balanced creativity
            response = await self.llm_client.generate(
                crossover_prompt,
                temperature=temperature,
                max_tokens=2500
            )

            # Parse response
            offspring1, offspring2 = self._parse_offspring_response(response)
            if offspring1 and offspring2:
                return offspring1, offspring2

        except Exception as e:
            logger.error(f"Semantic crossover failed: {e}")

        return self._simple_fallback_crossover(parent1, parent2)

    async def _multi_point_crossover(self, parent1: Idea, parent2: Idea, num_points: int = None, **kwargs) -> tuple[str, str]:
        """Multi-point crossover with intelligent split point selection."""

        # Split content into sentences for crossover points
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)

        if len(sentences1) < 2 or len(sentences2) < 2:
            return self._simple_fallback_crossover(parent1, parent2)

        # Determine number of crossover points
        if num_points is None:
            max_points = min(len(sentences1), len(sentences2)) - 1
            num_points = random.randint(1, min(3, max_points))

        # Select crossover points based on semantic boundaries
        points1 = self._select_crossover_points(sentences1, num_points)
        points2 = self._select_crossover_points(sentences2, num_points)

        # Perform crossover
        offspring1_segments = []
        offspring2_segments = []

        current_parent = 1  # Start with parent 1

        for i in range(len(points1) + 1):
            start1 = points1[i-1] if i > 0 else 0
            end1 = points1[i] if i < len(points1) else len(sentences1)

            start2 = points2[i-1] if i > 0 else 0
            end2 = points2[i] if i < len(points2) else len(sentences2)

            if current_parent == 1:
                offspring1_segments.extend(sentences1[start1:end1])
                offspring2_segments.extend(sentences2[start2:end2])
            else:
                offspring1_segments.extend(sentences2[start2:end2])
                offspring2_segments.extend(sentences1[start1:end1])

            current_parent = 3 - current_parent  # Switch between 1 and 2

        offspring1 = ' '.join(offspring1_segments)
        offspring2 = ' '.join(offspring2_segments)

        # Post-process for coherence if LLM is available
        if self.llm_client:
            try:
                offspring1 = await self._improve_coherence(offspring1)
                offspring2 = await self._improve_coherence(offspring2)
            except Exception as e:
                logger.warning(f"Coherence improvement failed: {e}")

        return offspring1, offspring2

    async def _uniform_crossover(self, parent1: Idea, parent2: Idea, mask_probability: float = 0.5, **kwargs) -> tuple[str, str]:
        """Uniform crossover with learned selection masks."""

        # Split into semantic units (sentences/phrases)
        units1 = self._split_into_semantic_units(parent1.content)
        units2 = self._split_into_semantic_units(parent2.content)

        # Create adaptive mask based on content quality indicators
        mask = self._create_adaptive_mask(units1, units2, parent1.fitness, parent2.fitness)

        # Apply uniform crossover
        offspring1_units = []
        offspring2_units = []

        max_units = max(len(units1), len(units2))

        for i in range(max_units):
            unit1 = units1[i] if i < len(units1) else ""
            unit2 = units2[i] if i < len(units2) else ""

            if i < len(mask) and mask[i]:
                offspring1_units.append(unit1)
                offspring2_units.append(unit2)
            else:
                offspring1_units.append(unit2)
                offspring2_units.append(unit1)

        offspring1 = ' '.join(filter(None, offspring1_units))
        offspring2 = ' '.join(filter(None, offspring2_units))

        return offspring1, offspring2

    async def _edge_recombination_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """Edge recombination preserving concept relationships."""

        # Extract concept graphs from both parents
        concepts1 = await self._extract_concept_graph(parent1.content)
        concepts2 = await self._extract_concept_graph(parent2.content)

        if not concepts1 or not concepts2:
            return await self._semantic_crossover(parent1, parent2)

        # Build adjacency information
        adjacency = self._build_concept_adjacency(concepts1, concepts2)

        # Generate offspring using edge recombination
        offspring1 = await self._generate_from_adjacency(adjacency, concepts1, concepts2, bias_toward=1)
        offspring2 = await self._generate_from_adjacency(adjacency, concepts1, concepts2, bias_toward=2)

        return offspring1, offspring2

    async def _order_based_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """Order-based crossover maintaining logical flow."""

        # Extract ordered elements (key phrases, steps, etc.)
        elements1 = self._extract_ordered_elements(parent1.content)
        elements2 = self._extract_ordered_elements(parent2.content)

        if len(elements1) < 2 or len(elements2) < 2:
            return await self._multi_point_crossover(parent1, parent2)

        # Select subset from each parent while preserving order
        subset_size = min(len(elements1), len(elements2)) // 2

        # For offspring 1: take ordered subset from parent1, fill gaps with parent2
        selected_indices1 = sorted(random.sample(range(len(elements1)), subset_size))
        selected_elements1 = [elements1[i] for i in selected_indices1]

        # Fill remaining positions with elements from parent2 (in order)
        remaining_elements2 = [elem for elem in elements2 if elem not in selected_elements1]

        # For offspring 2: reverse the process
        selected_indices2 = sorted(random.sample(range(len(elements2)), subset_size))
        selected_elements2 = [elements2[i] for i in selected_indices2]
        remaining_elements1 = [elem for elem in elements1 if elem not in selected_elements2]

        # Reconstruct offspring maintaining logical order
        offspring1 = await self._reconstruct_ordered_content(selected_elements1, remaining_elements2)
        offspring2 = await self._reconstruct_ordered_content(selected_elements2, remaining_elements1)

        return offspring1, offspring2

    async def _blend_crossover(self, parent1: Idea, parent2: Idea, alpha: float = None, **kwargs) -> tuple[str, str]:
        """Fitness-weighted blend crossover."""

        if alpha is None:
            # Adaptive alpha based on fitness difference
            fitness_diff = abs(parent1.fitness - parent2.fitness)
            alpha = min(0.5, fitness_diff)  # More blending when fitness is similar

        # Calculate blend weights based on fitness
        total_fitness = parent1.fitness + parent2.fitness + 1e-6  # Avoid division by zero
        weight1 = parent1.fitness / total_fitness
        weight2 = parent2.fitness / total_fitness

        if self.llm_client:
            # LLM-guided blending
            blend_prompt = f"""
            Create two offspring by blending these ideas with specified weights:

            IDEA 1 (Weight: {weight1:.2f}, Fitness: {parent1.fitness:.3f}):
            {parent1.content}

            IDEA 2 (Weight: {weight2:.2f}, Fitness: {parent2.fitness:.3f}):
            {parent2.content}

            OFFSPRING_1: Emphasize the higher-fitness parent ({weight1:.0%} from Idea 1, {weight2:.0%} from Idea 2)
            OFFSPRING_2: Create a balanced blend with variation (explore different emphasis patterns)

            Both offspring should be coherent, innovative, and incorporate elements proportional to parent fitness.

            OFFSPRING_1: [content]
            OFFSPRING_2: [content]
            """

            try:
                response = await self.llm_client.generate(
                    blend_prompt,
                    temperature=0.7,
                    max_tokens=2500
                )

                offspring1, offspring2 = self._parse_offspring_response(response)
                if offspring1 and offspring2:
                    return offspring1, offspring2

            except Exception as e:
                logger.error(f"Blend crossover failed: {e}")

        # Fallback to weighted sentence blending
        return self._weighted_sentence_blend(parent1, parent2, weight1, weight2)

    async def _concept_mapping_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """Concept mapping and recombination crossover."""

        if not self.llm_client:
            return await self._multi_point_crossover(parent1, parent2)

        # Extract and map concepts between parents
        concept_mapping_prompt = f"""
        Analyze these two ideas and identify key concepts that can be mapped and recombined:

        IDEA 1: {parent1.content}
        IDEA 2: {parent2.content}

        First, extract key concepts from each idea, then create two innovative offspring by:
        1. Mapping similar concepts between the ideas
        2. Combining complementary concepts
        3. Creating novel connections between previously unconnected concepts

        OFFSPRING_1: Focus on concept mapping - combine similar concepts in novel ways
        OFFSPRING_2: Focus on concept bridging - connect previously separate concepts

        OFFSPRING_1: [content]
        OFFSPRING_2: [content]
        """

        try:
            response = await self.llm_client.generate(
                concept_mapping_prompt,
                temperature=0.75,
                max_tokens=2500
            )

            offspring1, offspring2 = self._parse_offspring_response(response)
            if offspring1 and offspring2:
                return offspring1, offspring2

        except Exception as e:
            logger.error(f"Concept mapping crossover failed: {e}")

        return await self._semantic_crossover(parent1, parent2)

    async def _syntactic_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """Grammar-aware crossover preserving syntactic structure."""

        # Parse syntactic structures
        structure1 = self._analyze_syntactic_structure(parent1.content)
        structure2 = self._analyze_syntactic_structure(parent2.content)

        if self.llm_client:
            # LLM-guided syntactic crossover
            syntax_prompt = f"""
            Perform grammar-aware crossover of these ideas, preserving syntactic structure:

            IDEA 1 (Structure: {structure1}): {parent1.content}
            IDEA 2 (Structure: {structure2}): {parent2.content}

            Create offspring that:
            - Maintain grammatical correctness
            - Preserve the stronger syntactic patterns from each parent
            - Combine content while respecting sentence structure

            OFFSPRING_1: [content maintaining syntactic structure]
            OFFSPRING_2: [content maintaining syntactic structure]
            """

            try:
                response = await self.llm_client.generate(
                    syntax_prompt,
                    temperature=0.6,
                    max_tokens=2500
                )

                offspring1, offspring2 = self._parse_offspring_response(response)
                if offspring1 and offspring2:
                    return offspring1, offspring2

            except Exception as e:
                logger.error(f"Syntactic crossover failed: {e}")

        # Fallback to structure-aware simple crossover
        return self._structure_aware_crossover(parent1, parent2, structure1, structure2)

    async def _hierarchical_crossover(self, parent1: Idea, parent2: Idea, **kwargs) -> tuple[str, str]:
        """Hierarchical tree-based crossover for complex structures."""

        if not self.llm_client:
            return await self._multi_point_crossover(parent1, parent2)

        # Analyze hierarchical structure
        hierarchy_prompt = f"""
        Analyze the hierarchical structure of these ideas and perform tree-based crossover:

        IDEA 1: {parent1.content}
        IDEA 2: {parent2.content}

        Create offspring by:
        1. Identifying main themes and sub-themes in each idea
        2. Exchanging subtrees between the hierarchical structures
        3. Ensuring logical parent-child relationships are maintained

        OFFSPRING_1: Combine main structure of Idea 1 with subtrees from Idea 2
        OFFSPRING_2: Combine main structure of Idea 2 with subtrees from Idea 1

        OFFSPRING_1: [content]
        OFFSPRING_2: [content]
        """

        try:
            response = await self.llm_client.generate(
                hierarchy_prompt,
                temperature=0.7,
                max_tokens=2500
            )

            offspring1, offspring2 = self._parse_offspring_response(response)
            if offspring1 and offspring2:
                return offspring1, offspring2

        except Exception as e:
            logger.error(f"Hierarchical crossover failed: {e}")

        return await self._semantic_crossover(parent1, parent2)

    # Helper methods

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two content strings."""
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
            try:
                # Use TF-IDF vectors for similarity
                documents = [content1, content2]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                return similarity_matrix[0, 1]
            except Exception:
                pass

        # Fallback to simple word overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _estimate_content_complexity(self, content: str) -> float:
        """Estimate complexity of content based on various metrics."""
        if not content:
            return 0.0

        # Various complexity indicators
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_words_per_sentence = word_count / max(sentence_count, 1)

        # Vocabulary richness
        unique_words = len(set(content.lower().split()))
        vocabulary_richness = unique_words / max(word_count, 1)

        # Punctuation complexity
        punctuation_density = len(re.findall(r'[,.;:()[\]{}"]', content)) / max(len(content), 1)

        # Technical terms (simple heuristic)
        technical_terms = len(re.findall(r'\b(?:implement|algorithm|system|framework|architecture|methodology)\b', content.lower()))
        technical_density = technical_terms / max(word_count, 1)

        # Combine metrics (normalized to 0-1 range)
        complexity = (
            min(avg_words_per_sentence / 20, 1.0) * 0.3 +
            vocabulary_richness * 0.3 +
            min(punctuation_density * 100, 1.0) * 0.2 +
            min(technical_density * 10, 1.0) * 0.2
        )

        return complexity

    def _split_into_sentences(self, content: str) -> list[str]:
        """Split content into sentences."""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_semantic_units(self, content: str) -> list[str]:
        """Split content into semantic units (sentences, clauses, etc.)."""
        # More sophisticated splitting including clauses
        units = re.split(r'[.!?;,]+', content)
        return [unit.strip() for unit in units if unit.strip()]

    def _select_crossover_points(self, sentences: list[str], num_points: int) -> list[int]:
        """Select intelligent crossover points based on semantic boundaries."""
        if num_points >= len(sentences):
            return list(range(1, len(sentences)))

        # Simple approach: evenly spaced points
        # In a more sophisticated version, this could use semantic similarity
        step = len(sentences) / (num_points + 1)
        points = [int(step * (i + 1)) for i in range(num_points)]
        return sorted(set(points))  # Remove duplicates and sort

    def _create_adaptive_mask(self, units1: list[str], units2: list[str],
                             fitness1: float, fitness2: float) -> list[bool]:
        """Create adaptive mask based on content quality and fitness."""
        max_units = max(len(units1), len(units2))
        mask = []

        # Bias toward higher fitness parent
        fitness_bias = fitness1 / (fitness1 + fitness2 + 1e-6)

        for _i in range(max_units):
            # Random selection with fitness bias
            if random.random() < fitness_bias:
                mask.append(True)  # Select from parent 1
            else:
                mask.append(False)  # Select from parent 2

        return mask

    async def _extract_concept_graph(self, content: str) -> list[ConceptNode]:
        """Extract concept graph from content."""
        # Simple approach: key phrases as concepts
        # In practice, this could use NLP libraries for better concept extraction
        sentences = self._split_into_sentences(content)
        concepts = []

        for i, sentence in enumerate(sentences):
            # Extract key phrases (nouns and noun phrases)
            key_phrases = re.findall(r'\b(?:[A-Z][a-z]*\s*)+\b', sentence)
            for phrase in key_phrases:
                if len(phrase.strip()) > 3:  # Filter short phrases
                    concept = ConceptNode(phrase.strip(), importance=1.0, position=i)
                    concepts.append(concept)

        return concepts[:10]  # Limit to top concepts

    def _build_concept_adjacency(self, concepts1: list[ConceptNode],
                                concepts2: list[ConceptNode]) -> dict[str, list[str]]:
        """Build adjacency information for edge recombination."""
        adjacency = {}

        # Add adjacency from concepts1
        for i, concept in enumerate(concepts1):
            adjacent = []
            if i > 0:
                adjacent.append(concepts1[i-1].content)
            if i < len(concepts1) - 1:
                adjacent.append(concepts1[i+1].content)
            adjacency[concept.content] = adjacent

        # Add adjacency from concepts2
        for i, concept in enumerate(concepts2):
            if concept.content not in adjacency:
                adjacency[concept.content] = []

            if i > 0:
                adjacency[concept.content].append(concepts2[i-1].content)
            if i < len(concepts2) - 1:
                adjacency[concept.content].append(concepts2[i+1].content)

        return adjacency

    async def _generate_from_adjacency(self, adjacency: dict[str, list[str]],
                                      concepts1: list[ConceptNode],
                                      concepts2: list[ConceptNode],
                                      bias_toward: int) -> str:
        """Generate offspring content from adjacency information."""
        if not self.llm_client:
            # Simple fallback
            all_concepts = [c.content for c in concepts1 + concepts2]
            return ' '.join(random.sample(all_concepts, min(5, len(all_concepts))))

        # Use LLM to generate coherent content from concepts
        concepts_text = ', '.join(list(adjacency.keys()))

        prompt = f"""
        Generate a coherent idea using these extracted concepts and their relationships:
        Concepts: {concepts_text}

        Create a flowing, logical narrative that incorporates these concepts meaningfully.
        The result should be coherent and innovative.
        """

        try:
            response = await self.llm_client.generate(prompt, temperature=0.7, max_tokens=1500)
            return response.strip()
        except Exception:
            # Fallback
            selected_concepts = random.sample(list(adjacency.keys()), min(5, len(adjacency)))
            return '. '.join(selected_concepts) + '.'

    def _extract_ordered_elements(self, content: str) -> list[str]:
        """Extract ordered elements from content."""
        # Look for numbered lists, steps, or sequential elements

        # Try to find numbered items first
        numbered_items = re.findall(r'\d+[.)]\s*([^.]+)', content)
        if numbered_items:
            return numbered_items

        # Try bullet points
        bullet_items = re.findall(r'[•\-\*]\s*([^.]+)', content)
        if bullet_items:
            return bullet_items

        # Fall back to sentences
        return self._split_into_sentences(content)

    async def _reconstruct_ordered_content(self, primary_elements: list[str],
                                          secondary_elements: list[str]) -> str:
        """Reconstruct content maintaining logical order."""
        if not self.llm_client:
            # Simple concatenation
            all_elements = primary_elements + secondary_elements
            return '. '.join(all_elements) + '.'

        elements_text = '\n'.join([f"Primary: {elem}" for elem in primary_elements] +
                                 [f"Secondary: {elem}" for elem in secondary_elements])

        prompt = f"""
        Create a coherent idea by logically ordering and connecting these elements:

        {elements_text}

        Maintain logical flow and create smooth transitions between elements.
        """

        try:
            response = await self.llm_client.generate(prompt, temperature=0.6, max_tokens=1500)
            return response.strip()
        except Exception:
            return '. '.join(primary_elements + secondary_elements) + '.'

    def _weighted_sentence_blend(self, parent1: Idea, parent2: Idea,
                                weight1: float, weight2: float) -> tuple[str, str]:
        """Weighted blending of sentences as fallback."""
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)

        # Select sentences based on weights
        num_from_1 = int(len(sentences1) * weight1)
        num_from_2 = int(len(sentences2) * weight2)

        selected1 = random.sample(sentences1, min(num_from_1, len(sentences1)))
        selected2 = random.sample(sentences2, min(num_from_2, len(sentences2)))

        offspring1 = '. '.join(selected1 + selected2[:2]) + '.'
        offspring2 = '. '.join(selected2 + selected1[:2]) + '.'

        return offspring1, offspring2

    def _analyze_syntactic_structure(self, content: str) -> dict[str, Any]:
        """Analyze syntactic structure of content."""
        # Simple syntactic analysis
        sentences = self._split_into_sentences(content)

        structure = {
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
            "question_count": len(re.findall(r'\?', content)),
            "exclamation_count": len(re.findall(r'!', content)),
            "complex_sentences": len(re.findall(r'[,;:]', content))
        }

        return structure

    def _structure_aware_crossover(self, parent1: Idea, parent2: Idea,
                                  structure1: dict, structure2: dict) -> tuple[str, str]:
        """Structure-aware crossover fallback."""
        # Simple implementation: alternate sentences while trying to maintain structure
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)

        # Try to maintain similar sentence count and complexity
        target_length1 = int((structure1["sentence_count"] + structure2["sentence_count"]) / 2)
        target_length2 = target_length1

        offspring1_sentences = []
        offspring2_sentences = []

        for i in range(max(len(sentences1), len(sentences2))):
            if i < len(sentences1) and len(offspring1_sentences) < target_length1:
                offspring1_sentences.append(sentences1[i])
            if i < len(sentences2) and len(offspring2_sentences) < target_length2:
                offspring2_sentences.append(sentences2[i])

        offspring1 = '. '.join(offspring1_sentences) + '.'
        offspring2 = '. '.join(offspring2_sentences) + '.'

        return offspring1, offspring2

    async def _improve_coherence(self, content: str) -> str:
        """Improve coherence of generated content."""
        coherence_prompt = f"""
        Improve the coherence and flow of this text while preserving its core ideas:

        {content}

        Make it read naturally and logically while keeping the same key concepts.
        """

        try:
            response = await self.llm_client.generate(
                coherence_prompt,
                temperature=0.5,
                max_tokens=1500
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Coherence improvement failed: {e}")
            return content

    def _parse_offspring_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract offspring content."""
        lines = response.strip().split('\n')
        offspring1 = offspring2 = None

        for line in lines:
            line = line.strip()
            if line.startswith("OFFSPRING_1:"):
                offspring1 = line.replace("OFFSPRING_1:", "").strip()
            elif line.startswith("OFFSPRING_2:"):
                offspring2 = line.replace("OFFSPRING_2:", "").strip()

        return offspring1, offspring2

    def _simple_fallback_crossover(self, parent1: Idea, parent2: Idea) -> tuple[str, str]:
        """Simple fallback crossover when advanced methods fail."""
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)

        if not sentences1 or not sentences2:
            return parent1.content, parent2.content

        # Simple midpoint crossover
        mid1 = len(sentences1) // 2
        mid2 = len(sentences2) // 2

        offspring1 = '. '.join(sentences1[:mid1] + sentences2[mid2:]) + '.'
        offspring2 = '. '.join(sentences2[:mid2] + sentences1[mid1:]) + '.'

        return offspring1, offspring2

    # Performance tracking methods

    def record_fitness_improvement(self, operator: str, parent1_fitness: float,
                                  parent2_fitness: float, offspring_fitness: float):
        """Record fitness improvement for an operator."""
        avg_parent_fitness = (parent1_fitness + parent2_fitness) / 2
        improvement = offspring_fitness - avg_parent_fitness

        if operator in self.metrics.operator_fitness_improvements:
            self.metrics.operator_fitness_improvements[operator].append(improvement)
        else:
            self.metrics.operator_fitness_improvements[operator] = [improvement]

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "total_crossovers": sum(self.metrics.operator_usage.values()),
            "operator_performance": {}
        }

        for operator in CrossoverOperator:
            performance = self.metrics.get_operator_performance(operator.value)
            report["operator_performance"][operator.value] = performance

        # Add recommendations
        if report["total_crossovers"] > 10:
            best_operator = max(
                report["operator_performance"].items(),
                key=lambda x: x[1]["avg_fitness_improvement"]
            )
            report["recommended_operator"] = best_operator[0]

        return report

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = CrossoverMetrics()
        self.operator_weights = {op.value: 1.0 for op in CrossoverOperator}
