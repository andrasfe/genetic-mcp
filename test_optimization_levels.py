#!/usr/bin/env python3
"""Test genetic algorithm optimizations at different levels."""

import asyncio
import time
from typing import Any

from genetic_mcp.models import Idea, FitnessWeights
from genetic_mcp.optimization_config import OptimizationConfig
from genetic_mcp.genetic_algorithm_gpu_enhanced import GPUEnhancedGeneticAlgorithm, AdvancedGeneticParameters
from genetic_mcp.genetic_algorithm import GeneticAlgorithm
from genetic_mcp.models import GeneticParameters


async def test_optimization_level(level: str, population_size: int = 20) -> dict[str, Any]:
    """Test a specific optimization level."""
    print(f"\n{'='*60}")
    print(f"Testing optimization level: {level}")
    print(f"{'='*60}")
    
    # Create configuration
    config = OptimizationConfig(level=level)
    print(f"Configuration: {config.to_dict()}")
    
    # Create initial population
    initial_population = []
    for i in range(population_size):
        idea = Idea(
            id=f"initial_{i}",
            content=f"Initial idea {i}: A solution for optimizing performance",
            generation=0,
            fitness=0.0
        )
        initial_population.append(idea)
    
    target_prompt = "Generate innovative solutions for improving software performance"
    
    # Run evolution based on level
    start_time = time.time()
    
    if level == "basic":
        # Use basic genetic algorithm
        params = GeneticParameters(
            population_size=population_size,
            generations=3,
            mutation_rate=0.2,
            crossover_rate=0.7
        )
        ga = GeneticAlgorithm(parameters=params)
        
        # Mock fitness evaluation since we don't have LLM
        for idea in initial_population:
            idea.fitness = 0.5
            idea.scores = {'relevance': 0.5, 'novelty': 0.5, 'feasibility': 0.5}
        
        # Create next generations
        population = initial_population
        for gen in range(params.generations):
            new_pop = []
            for _ in range(params.population_size):
                parent1, parent2 = ga.select_parents(population)
                child_content = parent1.content + " evolved"
                child = Idea(
                    id=f"gen{gen}_child",
                    content=child_content,
                    generation=gen+1,
                    parent_ids=[parent1.id, parent2.id],
                    fitness=0.6,
                    scores={'relevance': 0.6, 'novelty': 0.6, 'feasibility': 0.6}
                )
                new_pop.append(child)
            population = new_pop
        
        final_population = population
        
    else:
        # Use GPU-enhanced algorithm for all other levels
        if config.use_gpu:
            from genetic_mcp.gpu_accelerated import GPUConfig
            gpu_config = GPUConfig(device="cpu")  # Force CPU for testing
        else:
            gpu_config = None
            
        params = AdvancedGeneticParameters(
            population_size=population_size,
            generations=3,
            selection_method="tournament" if level == "enhanced" else "adaptive",
            use_fitness_sharing=config.use_diversity_preservation,
            use_pareto_ranking=config.use_pareto_optimization,
            adaptive_mutation=config.use_adaptive_parameters,
            n_subpopulations=4 if level == "full" else 1
        )
        
        ga = GPUEnhancedGeneticAlgorithm(
            parameters=params,
            gpu_config=gpu_config
        )
        
        # Mock the evolution
        final_population = await ga.evolve_population(
            initial_population,
            target_prompt,
            generations=3
        )
    
    end_time = time.time()
    evolution_time = end_time - start_time
    
    # Calculate metrics
    avg_fitness = sum(idea.fitness for idea in final_population) / len(final_population)
    
    results = {
        "level": level,
        "population_size": population_size,
        "evolution_time": evolution_time,
        "avg_fitness": avg_fitness,
        "gpu_enabled": config.use_gpu,
        "adaptive_params": config.use_adaptive_parameters,
        "diversity_preservation": config.use_diversity_preservation,
        "final_population_size": len(final_population)
    }
    
    print(f"\nResults:")
    print(f"  Evolution time: {evolution_time:.3f}s")
    print(f"  Average fitness: {avg_fitness:.3f}")
    print(f"  Final population size: {len(final_population)}")
    
    return results


async def main():
    """Test all optimization levels."""
    print("Testing Genetic Algorithm Optimization Levels")
    print("=" * 60)
    
    levels = ["basic", "enhanced", "gpu", "full"]
    results = []
    
    for level in levels:
        try:
            result = await test_optimization_level(level)
            results.append(result)
        except Exception as e:
            print(f"\nError testing {level}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    print(f"{'Level':<10} {'Time (s)':<10} {'Avg Fitness':<12} {'GPU':<5} {'Adaptive':<9} {'Diversity':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['level']:<10} {r['evolution_time']:<10.3f} {r['avg_fitness']:<12.3f} "
              f"{str(r['gpu_enabled']):<5} {str(r['adaptive_params']):<9} {str(r['diversity_preservation']):<10}")


if __name__ == "__main__":
    asyncio.run(main())