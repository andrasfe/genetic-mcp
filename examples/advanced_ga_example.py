#!/usr/bin/env python3
"""Example demonstrating advanced genetic algorithm features."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_mcp.models import Idea, GeneticParameters, FitnessWeights
from genetic_mcp.genetic_algorithm import GeneticAlgorithm
from genetic_mcp.fitness import FitnessEvaluator
from genetic_mcp.diversity_manager import DiversityManager
import numpy as np


async def demonstrate_advanced_ga():
    """Demonstrate all advanced GA features."""
    
    print("=" * 60)
    print("Advanced Genetic Algorithm Demonstration")
    print("=" * 60)
    
    # Configure GA with adaptive parameters
    params = GeneticParameters(
        population_size=20,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_count=2
    )
    
    # Initialize components
    ga = GeneticAlgorithm(params)
    fitness_eval = FitnessEvaluator(FitnessWeights(
        relevance=0.4,
        novelty=0.3,
        feasibility=0.3
    ))
    diversity_mgr = DiversityManager()
    
    # Create initial population
    initial_ideas = [
        "AI-powered personal assistant for daily tasks",
        "Machine learning for predictive maintenance",
        "Natural language processing for customer service",
        "Computer vision for quality control",
        "Reinforcement learning for game AI",
        "Deep learning for medical diagnosis",
        "Neural networks for financial forecasting",
        "AI chatbot for education",
        "Machine learning for recommendation systems",
        "AI for autonomous vehicles",
        "NLP for sentiment analysis",
        "Computer vision for security",
        "AI for drug discovery",
        "Machine learning for climate prediction",
        "AI-powered content creation",
        "Neural networks for image generation",
        "AI for supply chain optimization",
        "Machine learning for fraud detection",
        "AI assistant for coding",
        "Deep learning for speech recognition"
    ]
    
    population = []
    embeddings = {}
    
    for i, content in enumerate(initial_ideas):
        idea = Idea(
            id=f"gen0_idea{i}",
            content=content,
            generation=0
        )
        population.append(idea)
        # Simulate embeddings (in real use, these would come from an embedding model)
        embeddings[idea.id] = np.random.randn(768)
    
    # Add embeddings to fitness evaluator
    for idea_id, embedding in embeddings.items():
        fitness_eval.add_embedding(idea_id, embedding)
    
    # Target embedding (what we're optimizing towards)
    target_embedding = np.random.randn(768)
    
    print(f"\nInitial population size: {len(population)}")
    
    # Evolution loop
    for generation in range(params.generations):
        print(f"\n--- Generation {generation + 1} ---")
        
        # 1. Evaluate fitness
        print("1. Evaluating fitness...")
        fitness_eval.evaluate_population(population, target_embedding)
        
        # 2. Calculate diversity metrics
        print("2. Calculating diversity...")
        diversity_metrics = diversity_mgr.calculate_diversity_metrics(population, embeddings)
        print(f"   Simpson diversity: {diversity_metrics['simpson_diversity']:.3f}")
        print(f"   Average distance: {diversity_metrics['average_distance']:.3f}")
        
        # 3. Apply fitness sharing if diversity is low
        if diversity_metrics['simpson_diversity'] < 0.5:
            print("3. Applying fitness sharing (low diversity detected)...")
            shared_fitness = ga.calculate_fitness_sharing(population)
            probabilities = fitness_eval.get_selection_probabilities_shared(population, shared_fitness)
        else:
            probabilities = fitness_eval.get_selection_probabilities(population)
        
        # 4. Check for convergence
        if ga.check_convergence(population):
            print("\n✓ Convergence detected! Stopping evolution.")
            break
        
        # 5. Adaptively choose selection method
        selection_method = ga.get_selection_method_for_generation(generation)
        print(f"4. Using {selection_method} selection for this generation")
        
        # 6. Create next generation with advanced features
        use_crowding = diversity_metrics['simpson_diversity'] < 0.4
        new_population = ga.create_next_generation(
            population,
            probabilities,
            generation + 1,
            selection_method=selection_method,
            use_fitness_sharing=diversity_metrics['simpson_diversity'] < 0.5,
            use_crowding=use_crowding
        )
        
        # 7. Generate embeddings for new ideas (simulate)
        for idea in new_population:
            if idea.id not in embeddings:
                # In real use, this would generate actual embeddings
                embeddings[idea.id] = np.random.randn(768)
                fitness_eval.add_embedding(idea.id, embeddings[idea.id])
        
        # 8. Apply species-based evolution if needed
        if generation % 3 == 0:
            print("5. Performing speciation...")
            species = diversity_mgr.apply_speciation(new_population, embeddings)
            print(f"   Found {len(species)} species")
            species_stats = diversity_mgr.get_species_statistics()
            print(f"   Singleton species: {species_stats['singleton_species']}")
        
        # 9. Show adaptive parameters
        print("6. Adaptive parameters:")
        print(f"   Mutation rate: {ga.parameters.mutation_rate:.3f}")
        print(f"   Crossover rate: {ga.parameters.crossover_rate:.3f}")
        print(f"   Elitism count: {ga.parameters.elitism_count}")
        print(f"   Boltzmann temperature: {ga.boltzmann_temperature:.3f}")
        
        population = new_population
        
        # 10. Report best idea
        best_idea = max(population, key=lambda x: x.fitness)
        print(f"\n   Best fitness: {best_idea.fitness:.3f}")
        print(f"   Best idea: {best_idea.content[:50]}...")
    
    # Final results
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    
    # Evaluate final population
    fitness_eval.evaluate_population(population, target_embedding)
    
    # Get top 5 ideas
    top_ideas = sorted(population, key=lambda x: x.fitness, reverse=True)[:5]
    
    print("\nTop 5 Ideas:")
    for i, idea in enumerate(top_ideas, 1):
        print(f"\n{i}. {idea.content}")
        print(f"   Fitness: {idea.fitness:.3f}")
        print(f"   Scores: R={idea.scores.get('relevance', 0):.3f}, "
              f"N={idea.scores.get('novelty', 0):.3f}, "
              f"F={idea.scores.get('feasibility', 0):.3f}")
    
    # Show diversity preservation results
    final_diversity = diversity_mgr.calculate_diversity_metrics(population, embeddings)
    print(f"\nFinal diversity metrics:")
    print(f"  Simpson diversity: {final_diversity['simpson_diversity']:.3f}")
    print(f"  Shannon diversity: {final_diversity['shannon_diversity']:.3f}")
    print(f"  Average distance: {final_diversity['average_distance']:.3f}")
    print(f"  Coverage: {final_diversity['coverage']:.3f}")
    print(f"  Evenness: {final_diversity['evenness']:.3f}")
    
    # Test Pareto ranking
    print("\n--- Testing Pareto Ranking ---")
    pareto_ranks = fitness_eval.calculate_pareto_ranks(population)
    rank_counts = {}
    for idea_id, rank in pareto_ranks.items():
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    print("Pareto rank distribution:")
    for rank in sorted(rank_counts.keys()):
        print(f"  Rank {rank}: {rank_counts[rank]} ideas")
    
    # Show convergence history
    print(f"\nConvergence metrics:")
    print(f"  Generations run: {len(ga.fitness_history)}")
    print(f"  Final stagnation counter: {ga.stagnation_counter}")
    print(f"  Fitness progression: {[f'{f:.3f}' for f in ga.fitness_history[-5:]]}")
    
    print("\n✓ Advanced GA demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_ga())