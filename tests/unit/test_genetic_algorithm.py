"""Unit tests for genetic algorithm."""

import pytest
from genetic_mcp.models import Idea, GeneticParameters
from genetic_mcp.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm:
    """Test the GeneticAlgorithm class."""
    
    def test_algorithm_creation(self):
        """Test creating genetic algorithm."""
        ga = GeneticAlgorithm()
        
        assert ga.parameters.population_size == 10
        assert ga.parameters.mutation_rate == 0.1
        assert ga.parameters.crossover_rate == 0.7
    
    def test_algorithm_with_custom_parameters(self):
        """Test algorithm with custom parameters."""
        params = GeneticParameters(
            population_size=20,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        ga = GeneticAlgorithm(params)
        
        assert ga.parameters.population_size == 20
        assert ga.parameters.mutation_rate == 0.2
        assert ga.parameters.crossover_rate == 0.8
    
    def test_select_parents(self):
        """Test parent selection."""
        ga = GeneticAlgorithm()
        
        population = [
            Idea(id="1", content="A", fitness=0.1),
            Idea(id="2", content="B", fitness=0.3),
            Idea(id="3", content="C", fitness=0.5),
            Idea(id="4", content="D", fitness=0.1),
        ]
        
        probabilities = [0.1, 0.3, 0.5, 0.1]
        
        parent1, parent2 = ga.select_parents(population, probabilities)
        
        assert parent1 in population
        assert parent2 in population
        assert parent1.id != parent2.id
    
    def test_crossover(self):
        """Test crossover operation."""
        ga = GeneticAlgorithm()
        
        parent1 = Idea(
            id="p1",
            content="This is the first parent. It has multiple sentences. Each one is important."
        )
        parent2 = Idea(
            id="p2",
            content="Second parent here. Different content entirely. With unique ideas."
        )
        
        offspring1, offspring2 = ga.crossover(parent1, parent2)
        
        # Offspring should be different from parents (usually)
        assert isinstance(offspring1, str)
        assert isinstance(offspring2, str)
        assert len(offspring1) > 0
        assert len(offspring2) > 0
    
    def test_mutate_operations(self):
        """Test mutation operations."""
        ga = GeneticAlgorithm()
        
        original = "This is a test idea with some content for mutation testing."
        
        # Test different mutation types
        mutated_rephrase = ga._rephrase_mutation(original)
        mutated_add = ga._add_mutation(original)
        mutated_remove = ga._remove_mutation(original)
        mutated_modify = ga._modify_mutation(original)
        
        # Mutations should produce valid strings
        assert isinstance(mutated_rephrase, str)
        assert isinstance(mutated_add, str)
        assert isinstance(mutated_remove, str)
        assert isinstance(mutated_modify, str)
        
        # Add mutation should make content longer
        assert len(mutated_add) > len(original)
    
    def test_mutate_with_probability(self):
        """Test mutation with probability."""
        # High mutation rate
        params = GeneticParameters(mutation_rate=1.0)
        ga = GeneticAlgorithm(params)
        
        original = "This is a test idea."
        mutated = ga.mutate(original)
        
        # With 100% mutation rate, content should change
        assert mutated != original
        
        # Low mutation rate
        params2 = GeneticParameters(mutation_rate=0.0)
        ga2 = GeneticAlgorithm(params2)
        
        not_mutated = ga2.mutate(original)
        
        # With 0% mutation rate, content should not change
        assert not_mutated == original
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        ga = GeneticAlgorithm()
        
        # Test with periods
        content1 = "First sentence. Second sentence. Third sentence."
        sentences1 = ga._extract_sentences(content1)
        assert len(sentences1) == 3
        
        # Test with bullet points
        content2 = "• First point\n• Second point\n• Third point"
        sentences2 = ga._extract_sentences(content2)
        assert len(sentences2) >= 3
        
        # Test with single sentence
        content3 = "Single sentence without breaks"
        sentences3 = ga._extract_sentences(content3)
        assert len(sentences3) >= 1
    
    def test_create_next_generation(self):
        """Test creating next generation."""
        params = GeneticParameters(
            population_size=4,
            elitism_count=1
        )
        ga = GeneticAlgorithm(params)
        
        population = [
            Idea(id="1", content="First idea", fitness=0.9),
            Idea(id="2", content="Second idea", fitness=0.5),
            Idea(id="3", content="Third idea", fitness=0.3),
            Idea(id="4", content="Fourth idea", fitness=0.1),
        ]
        
        probabilities = [0.4, 0.3, 0.2, 0.1]
        
        new_generation = ga.create_next_generation(population, probabilities, generation=1)
        
        assert len(new_generation) == params.population_size
        
        # Check elitism - best idea should be preserved
        elite_contents = [idea.content for idea in new_generation if idea.metadata.get("elite")]
        assert len(elite_contents) >= params.elitism_count
        
        # All new ideas should have correct generation
        for idea in new_generation:
            assert idea.generation == 1
            assert idea.id.startswith("gen1_")
    
    def test_crossover_edge_cases(self):
        """Test crossover with edge cases."""
        ga = GeneticAlgorithm()
        
        # Single sentence parents
        parent1 = Idea(id="p1", content="Single sentence")
        parent2 = Idea(id="p2", content="Another sentence")
        
        offspring1, offspring2 = ga.crossover(parent1, parent2)
        
        assert len(offspring1) > 0
        assert len(offspring2) > 0
        
        # Empty content (should not happen but test anyway)
        parent3 = Idea(id="p3", content="")
        parent4 = Idea(id="p4", content="Some content")
        
        offspring3, offspring4 = ga.crossover(parent3, parent4)
        
        # Should handle gracefully
        assert isinstance(offspring3, str)
        assert isinstance(offspring4, str)