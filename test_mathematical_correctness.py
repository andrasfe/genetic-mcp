#!/usr/bin/env python3
"""Test mathematical correctness of genetic algorithm implementations."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_mcp.genetic_algorithm import GeneticAlgorithm
from genetic_mcp.models import Idea, GeneticParameters
from genetic_mcp.fitness import FitnessEvaluator


def test_selection_probabilities():
    """Test that all selection methods produce valid probability distributions."""
    print("Testing selection method probability distributions...")
    
    # Create test population
    population = []
    for i in range(10):
        idea = Idea(
            id=f"idea_{i}",
            content=f"Test idea {i}",
            generation=0,
            fitness=np.random.random()
        )
        population.append(idea)
    
    ga = GeneticAlgorithm()
    
    # Test each selection method
    methods = ["roulette", "tournament", "boltzmann", "sus", "rank"]
    
    for method in methods:
        print(f"\nTesting {method} selection:")
        
        # For methods that don't use probabilities directly
        if method in ["tournament"]:
            # Just test that it returns valid parents
            try:
                parent1, parent2 = ga.select_parents(population, [], method=method)
                assert parent1 in population
                assert parent2 in population
                assert parent1.id != parent2.id
                print(f"  ✓ {method} returns valid parents")
            except Exception as e:
                print(f"  ✗ {method} failed: {e}")
            continue
        
        # For probability-based methods
        try:
            # Create probabilities for roulette wheel
            fitness_eval = FitnessEvaluator()
            probabilities = fitness_eval.get_selection_probabilities(population)
            
            # Test selection
            parent1, parent2 = ga.select_parents(population, probabilities, method=method)
            
            # Verify parents are from population
            assert parent1 in population
            assert parent2 in population
            assert parent1.id != parent2.id
            
            print(f"  ✓ {method} selection works correctly")
            
        except Exception as e:
            print(f"  ✗ {method} selection failed: {e}")


def test_boltzmann_numerical_stability():
    """Test Boltzmann selection with extreme fitness values."""
    print("\nTesting Boltzmann selection numerical stability...")
    
    # Test with very large fitness values
    population = []
    fitness_values = [1000, 2000, 3000, 4000, 5000]  # Large values
    
    for i, fitness in enumerate(fitness_values):
        idea = Idea(
            id=f"idea_{i}",
            content=f"Test idea {i}",
            generation=0,
            fitness=fitness
        )
        population.append(idea)
    
    ga = GeneticAlgorithm()
    
    try:
        parent1, parent2 = ga._boltzmann_selection(population)
        print("  ✓ Handles large fitness values")
    except Exception as e:
        print(f"  ✗ Failed with large values: {e}")
    
    # Test with negative fitness values
    population_neg = []
    fitness_values_neg = [-10, -5, 0, 5, 10]
    
    for i, fitness in enumerate(fitness_values_neg):
        idea = Idea(
            id=f"idea_neg_{i}",
            content=f"Test idea {i}",
            generation=0,
            fitness=fitness
        )
        population_neg.append(idea)
    
    try:
        parent1, parent2 = ga._boltzmann_selection(population_neg)
        print("  ✓ Handles negative fitness values")
    except Exception as e:
        print(f"  ✗ Failed with negative values: {e}")


def test_rank_selection_probabilities():
    """Test that rank-based selection produces correct probability distribution."""
    print("\nTesting rank-based selection probability distribution...")
    
    # Create population with known fitness order
    population = []
    for i in range(5):
        idea = Idea(
            id=f"idea_{i}",
            content=f"Test idea {i}",
            generation=0,
            fitness=float(i)  # 0, 1, 2, 3, 4
        )
        population.append(idea)
    
    ga = GeneticAlgorithm()
    ga.rank_selection_pressure = 2.0  # Maximum pressure
    
    # Test selection multiple times to verify distribution
    selection_counts = {f"idea_{i}": 0 for i in range(5)}
    
    for _ in range(1000):
        parent1, parent2 = ga._rank_based_selection(population)
        selection_counts[parent1.id] += 1
        selection_counts[parent2.id] += 1
    
    # Higher fitness should be selected more often
    counts = [selection_counts[f"idea_{i}"] for i in range(5)]
    
    # Check if selection frequency increases with rank
    is_increasing = all(counts[i] <= counts[i+1] for i in range(4))
    
    if is_increasing:
        print("  ✓ Selection frequency increases with fitness rank")
    else:
        print("  ✗ Selection frequency does not follow rank order")
    
    print(f"  Selection counts: {counts}")


def test_fitness_sharing():
    """Test fitness sharing calculations."""
    print("\nTesting fitness sharing...")
    
    # Create population with some similar ideas
    population = []
    contents = [
        "Machine learning for prediction",
        "Machine learning for classification",  # Similar to first
        "Deep neural networks",
        "Reinforcement learning agents",
        "Machine learning for prediction tasks"  # Very similar to first
    ]
    
    for i, content in enumerate(contents):
        idea = Idea(
            id=f"idea_{i}",
            content=content,
            generation=0,
            fitness=1.0  # Same fitness for all
        )
        population.append(idea)
    
    ga = GeneticAlgorithm()
    shared_fitness = ga.calculate_fitness_sharing(population, sigma_share=0.5)
    
    # Ideas 0 and 4 should have lower shared fitness (they're similar)
    # Ideas 0, 1, and 4 form a niche
    print(f"  Original fitness: {[idea.fitness for idea in population]}")
    print(f"  Shared fitness: {[shared_fitness[idea.id] for idea in population]}")
    
    # Verify that similar ideas have reduced fitness
    if shared_fitness["idea_0"] < 1.0 and shared_fitness["idea_4"] < 1.0:
        print("  ✓ Similar ideas have reduced shared fitness")
    else:
        print("  ✗ Fitness sharing not working correctly")


def test_convergence_detection():
    """Test convergence detection criteria."""
    print("\nTesting convergence detection...")
    
    ga = GeneticAlgorithm()
    
    # Simulate converged population (low diversity, plateau)
    ga.fitness_history = [0.8, 0.81, 0.82, 0.82, 0.82, 0.82]
    ga.diversity_history = [0.5, 0.4, 0.3, 0.2, 0.15, 0.15]
    
    # Create uniform population
    population = []
    for i in range(10):
        idea = Idea(
            id=f"idea_{i}",
            content="Very similar content with minor variation " + str(i),
            generation=5,
            fitness=0.82 + np.random.normal(0, 0.01)  # Small variance
        )
        population.append(idea)
    
    converged = ga.check_convergence(population, window_size=3)
    
    if converged:
        print("  ✓ Correctly detected convergence")
    else:
        print("  ✗ Failed to detect convergence")
    
    # Test non-converged population
    ga.fitness_history = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    ga.diversity_history = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    
    converged = ga.check_convergence(population, window_size=3)
    
    if not converged:
        print("  ✓ Correctly detected non-convergence")
    else:
        print("  ✗ Incorrectly detected convergence")


def test_adaptive_parameters():
    """Test adaptive parameter updates."""
    print("\nTesting adaptive parameter control...")
    
    ga = GeneticAlgorithm(GeneticParameters(
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_count=2
    ))
    
    # Create diverse population
    population = []
    for i in range(10):
        idea = Idea(
            id=f"idea_{i}",
            content=f"Unique idea number {i} with different content",
            generation=0,
            fitness=np.random.random()
        )
        population.append(idea)
    
    # Test parameter adaptation
    initial_mutation = ga.parameters.mutation_rate
    initial_crossover = ga.parameters.crossover_rate
    
    # Update with high diversity
    ga.update_adaptive_parameters(population)
    
    print(f"  Initial mutation rate: {initial_mutation}")
    print(f"  Adapted mutation rate: {ga.parameters.mutation_rate}")
    
    # Simulate low diversity
    for idea in population:
        idea.content = "Same content for all"
    
    ga.update_adaptive_parameters(population)
    
    if ga.parameters.mutation_rate > initial_mutation:
        print("  ✓ Mutation rate increased with low diversity")
    else:
        print("  ✗ Mutation rate did not increase appropriately")


def test_crowding_distance():
    """Test crowding distance calculation."""
    print("\nTesting crowding distance calculation...")
    
    # Create population with known objective values
    population = []
    objectives_data = [
        {"relevance": 0.1, "novelty": 0.1, "feasibility": 0.1},  # Corner point
        {"relevance": 0.5, "novelty": 0.5, "feasibility": 0.5},  # Middle point
        {"relevance": 0.9, "novelty": 0.9, "feasibility": 0.9},  # Corner point
        {"relevance": 0.4, "novelty": 0.5, "feasibility": 0.6},  # Close to middle
        {"relevance": 0.6, "novelty": 0.5, "feasibility": 0.4},  # Close to middle
    ]
    
    for i, scores in enumerate(objectives_data):
        idea = Idea(
            id=f"idea_{i}",
            content=f"Test idea {i}",
            generation=0,
            fitness=sum(scores.values()) / 3,
            scores=scores
        )
        population.append(idea)
    
    ga = GeneticAlgorithm()
    crowding_distances = ga.calculate_crowding_distance(population)
    
    print("  Crowding distances:")
    for idea in population:
        dist = crowding_distances[idea.id]
        print(f"    {idea.id}: {dist:.3f} (scores: {idea.scores})")
    
    # Corner points should have infinite distance
    if (crowding_distances["idea_0"] == float('inf') and 
        crowding_distances["idea_2"] == float('inf')):
        print("  ✓ Boundary points have infinite crowding distance")
    else:
        print("  ✗ Boundary points do not have correct crowding distance")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Mathematical Correctness of Genetic Algorithm")
    print("=" * 60)
    
    test_selection_probabilities()
    test_boltzmann_numerical_stability()
    test_rank_selection_probabilities()
    test_fitness_sharing()
    test_convergence_detection()
    test_adaptive_parameters()
    test_crowding_distance()
    
    print("\n" + "=" * 60)
    print("Mathematical correctness tests completed!")
    print("=" * 60)