"""Test script for optimized genetic algorithm."""

import asyncio
import numpy as np
from datetime import datetime
import json

from genetic_mcp.models import (
    Idea, GeneticParameters, FitnessWeights
)
from genetic_mcp.genetic_algorithm import GeneticAlgorithm
from genetic_mcp.genetic_algorithm_optimized import OptimizedGeneticAlgorithm
from genetic_mcp.fitness import FitnessEvaluator
from genetic_mcp.fitness_enhanced import EnhancedFitnessEvaluator
from genetic_mcp.optimization_coordinator import OptimizationCoordinator, OptimizationConfig


async def compare_algorithms():
    """Compare original and optimized genetic algorithms."""
    
    # Test parameters
    test_prompt = "Design an innovative mobile app for sustainable living"
    population_size = 20
    generations = 10
    
    # Common parameters
    parameters = GeneticParameters(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism_count=2
    )
    
    weights = FitnessWeights(
        relevance=0.4,
        novelty=0.3,
        feasibility=0.3
    )
    
    # Generate initial population
    initial_ideas = []
    for i in range(population_size):
        idea = Idea(
            id=f"initial_{i}",
            content=f"Mobile app idea {i}: Focus on {['recycling', 'energy', 'transport', 'food waste', 'water'][i % 5]} "
                   f"with features like {['tracking', 'gamification', 'social', 'rewards', 'education'][i % 5]}",
            generation=0
        )
        initial_ideas.append(idea)
    
    # Target embedding (simplified)
    target_embedding = np.random.randn(768)
    
    print("=" * 80)
    print("GENETIC ALGORITHM COMPARISON TEST")
    print("=" * 80)
    print(f"Prompt: {test_prompt}")
    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")
    print()
    
    # Test 1: Original Algorithm
    print("Testing Original Algorithm...")
    start_time = datetime.now()
    
    original_ga = GeneticAlgorithm(parameters)
    original_evaluator = FitnessEvaluator(weights)
    
    # Add mock embeddings
    for idea in initial_ideas:
        original_evaluator.add_embedding(idea.id, np.random.randn(768).tolist())
    
    current_pop = initial_ideas.copy()
    
    for gen in range(generations):
        # Evaluate fitness
        original_evaluator.evaluate_population(current_pop, target_embedding.tolist())
        
        # Get selection probabilities
        probs = original_evaluator.get_selection_probabilities(current_pop)
        
        # Create next generation
        current_pop = original_ga.create_next_generation(current_pop, probs, gen + 1)
        
        # Add embeddings for new population
        for idea in current_pop:
            if idea.id not in original_evaluator.embeddings_cache:
                original_evaluator.add_embedding(idea.id, np.random.randn(768).tolist())
    
    # Final evaluation
    original_evaluator.evaluate_population(current_pop, target_embedding.tolist())
    original_time = (datetime.now() - start_time).total_seconds()
    
    original_best = max(current_pop, key=lambda x: x.fitness)
    original_avg_fitness = sum(idea.fitness for idea in current_pop) / len(current_pop)
    
    print(f"Original Algorithm Complete!")
    print(f"Time: {original_time:.2f} seconds")
    print(f"Best Fitness: {original_best.fitness:.3f}")
    print(f"Average Fitness: {original_avg_fitness:.3f}")
    print(f"Best Idea: {original_best.content[:100]}...")
    print()
    
    # Test 2: Optimized Algorithm without LLM
    print("Testing Optimized Algorithm (without LLM)...")
    start_time = datetime.now()
    
    config = OptimizationConfig(
        use_llm_operators=False,  # No LLM for fair comparison
        use_adaptive_parameters=True,
        use_diversity_preservation=True,
        use_pareto_optimization=True,
        use_early_stopping=True
    )
    
    # Create mock session
    from genetic_mcp.models import Session, EvolutionMode
    session = Session(
        id="test_session",
        client_id="test_client",
        prompt=test_prompt,
        mode=EvolutionMode.ITERATIVE,
        parameters=parameters,
        fitness_weights=weights
    )
    
    coordinator = OptimizationCoordinator(
        parameters=parameters,
        fitness_weights=weights,
        llm_client=None,
        config=config
    )
    
    # Run optimized evolution
    top_ideas, metadata = await coordinator.run_evolution(
        initial_ideas.copy(),
        test_prompt,
        target_embedding,
        session
    )
    
    optimized_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Optimized Algorithm Complete!")
    print(f"Time: {optimized_time:.2f} seconds")
    print(f"Generations Run: {metadata['total_generations']}")
    print(f"Converged At: {metadata.get('converged_at', 'Did not converge')}")
    print(f"Best Fitness: {top_ideas[0].fitness:.3f}")
    print(f"Best Idea: {top_ideas[0].content[:100]}...")
    print()
    
    # Print detailed comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    improvement = ((top_ideas[0].fitness - original_best.fitness) / original_best.fitness) * 100
    print(f"Fitness Improvement: {improvement:+.1f}%")
    
    time_ratio = optimized_time / original_time
    print(f"Time Ratio: {time_ratio:.2f}x")
    
    print("\nDiversity Metrics (Final Population):")
    for metric, value in metadata['final_diversity'].items():
        print(f"  {metric}: {value:.3f}")
    
    print("\nSpecies Statistics:")
    species_stats = metadata.get('species_statistics', {})
    print(f"  Total Species: {species_stats.get('total_species', 0)}")
    print(f"  Singleton Species: {species_stats.get('singleton_species', 0)}")
    
    print("\nOptimization Report:")
    report = coordinator.get_optimization_report()
    
    print(f"  Selection Methods Used: {set(p['selection_method'] for p in report['parameter_adaptation'])}")
    print(f"  Average Computation Time per Generation: {report['performance']['total_computation_time'] / report['performance']['total_generations']:.2f}s")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        # Fitness progression plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        generations_x = [s['generation'] for s in report['fitness_progression']]
        best_fitness = [s['best'] for s in report['fitness_progression']]
        avg_fitness = [s['average'] for s in report['fitness_progression']]
        
        plt.plot(generations_x, best_fitness, 'b-', label='Best Fitness')
        plt.plot(generations_x, avg_fitness, 'g--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Progression')
        plt.legend()
        plt.grid(True)
        
        # Diversity progression plot
        plt.subplot(1, 2, 2)
        simpson_div = [s['simpson'] for s in report['diversity_progression']]
        
        plt.plot(generations_x, simpson_div, 'r-', label='Simpson Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Diversity Progression')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_comparison.png')
        print("\nPlots saved to optimization_comparison.png")
        
    except ImportError:
        print("\n(Matplotlib not available for plotting)")
    
    # Save detailed report
    with open('optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("\nDetailed report saved to optimization_report.json")


async def test_selection_strategies():
    """Test different selection strategies."""
    print("\n" + "=" * 80)
    print("SELECTION STRATEGY COMPARISON")
    print("=" * 80)
    
    # Create test population
    test_pop = []
    for i in range(10):
        idea = Idea(
            id=f"test_{i}",
            content=f"Test idea {i}",
            fitness=i / 10.0  # Linear fitness distribution
        )
        test_pop.append(idea)
    
    ga = OptimizedGeneticAlgorithm()
    
    strategies = ["tournament", "boltzmann", "sus", "rank"]
    results = {}
    
    for strategy in strategies:
        selected_counts = {idea.id: 0 for idea in test_pop}
        
        # Run selection multiple times
        for _ in range(1000):
            parent1, parent2 = await ga.select_parents_advanced(test_pop, strategy)
            selected_counts[parent1.id] += 1
            selected_counts[parent2.id] += 1
        
        # Calculate selection probabilities
        total_selections = sum(selected_counts.values())
        selection_probs = {
            idea_id: count / total_selections 
            for idea_id, count in selected_counts.items()
        }
        
        results[strategy] = selection_probs
        
        print(f"\n{strategy.upper()} Selection:")
        for idea in sorted(test_pop, key=lambda x: x.fitness, reverse=True):
            prob = selection_probs[idea.id]
            print(f"  {idea.id} (fitness={idea.fitness:.1f}): {prob:.3f}")
    
    # Compare selection pressure
    print("\nSelection Pressure Analysis:")
    for strategy in strategies:
        probs = list(results[strategy].values())
        pressure = max(probs) / min(probs) if min(probs) > 0 else float('inf')
        print(f"  {strategy}: {pressure:.2f}")


if __name__ == "__main__":
    print("Starting Genetic Algorithm Optimization Tests...")
    
    # Run comparison
    asyncio.run(compare_algorithms())
    
    # Test selection strategies
    asyncio.run(test_selection_strategies())
    
    print("\nAll tests complete!")