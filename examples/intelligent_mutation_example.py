#!/usr/bin/env python3
"""Example demonstrating intelligent mutation strategies in genetic-mcp."""

import asyncio
import logging
import os
from typing import List

# Set up the Python path to import genetic_mcp modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_mcp.genetic_algorithm_optimized import OptimizedGeneticAlgorithm
from genetic_mcp.intelligent_mutation import IntelligentMutationManager, MutationStrategy
from genetic_mcp.llm_client import LLMClient
from genetic_mcp.models import GeneticParameters, Idea
from genetic_mcp.fitness import FitnessEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for demonstration purposes."""
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """Generate a mock response based on the prompt type."""
        prompt_lower = prompt.lower()
        
        if "slightly modify" in prompt_lower or "small improvement" in prompt_lower:
            return "An enhanced system for automated task management using advanced AI algorithms with machine learning capabilities"
        elif "creative twist" in prompt_lower:
            return "A gamified system for automated task management using AI algorithms that rewards users for completing tasks efficiently"
        elif "significantly transform" in prompt_lower or "disruptive" in prompt_lower:
            return "A distributed blockchain-based system for decentralized task management using smart contracts and AI consensus mechanisms"
        elif "component" in prompt_lower:
            return "A system for automated task management using AI algorithms with improved natural language processing and predictive analytics"
        elif "memetic" in prompt_lower or "local search" in prompt_lower:
            return "A refined system for automated task management using optimized AI algorithms with enhanced performance metrics"
        elif "gradient" in prompt_lower or "direction" in prompt_lower:
            return "A system for automated task management using AI algorithms optimized for scalability and user experience"
        elif "temperature" in prompt_lower:
            if "high temperature" in prompt_lower:
                return "A revolutionary quantum-powered system for automated task management using next-generation AI algorithms"
            else:
                return "A system for automated task management using well-established AI algorithms with proven reliability"
        else:
            return "A modified system for automated task management using AI algorithms with additional features"


async def demonstrate_intelligent_mutation():
    """Demonstrate intelligent mutation strategies."""
    print("=== Intelligent Mutation Demonstration ===\n")
    
    # Create mock LLM client
    llm_client = MockLLMClient()
    
    # Create initial population
    initial_ideas = [
        Idea(
            id=f"idea_{i}",
            content=f"A system for automated task management using AI algorithms - variation {i}",
            generation=0,
            fitness=0.5 + (i * 0.05)  # Varying fitness levels
        )
        for i in range(5)
    ]
    
    # Create fitness evaluator (with dummy embeddings)
    fitness_evaluator = FitnessEvaluator()
    target_embedding = [0.1] * 384  # Mock target embedding
    
    # Add mock embeddings for ideas
    for idea in initial_ideas:
        mock_embedding = [0.1 + (i * 0.01) for i in range(384)]
        fitness_evaluator.add_embedding(idea.id, mock_embedding)
    
    # Evaluate initial fitness
    fitness_evaluator.evaluate_population(initial_ideas, target_embedding)
    
    print("Initial Population:")
    for idea in initial_ideas:
        print(f"  {idea.id}: fitness={idea.fitness:.3f}")
        print(f"    Content: {idea.content}\n")
    
    # Create intelligent mutation manager
    mutation_manager = IntelligentMutationManager(llm_client=llm_client)
    
    print("=== Testing Different Mutation Strategies ===\n")
    
    # Test each mutation strategy
    strategies_to_test = [
        MutationStrategy.RANDOM,
        MutationStrategy.GUIDED,
        MutationStrategy.ADAPTIVE,
        MutationStrategy.MEMETIC,
        MutationStrategy.CONTEXT_AWARE,
        MutationStrategy.COMPONENT_BASED,
        MutationStrategy.HILL_CLIMBING,
        MutationStrategy.SIMULATED_ANNEALING,
        MutationStrategy.GRADIENT_BASED
    ]
    
    mutated_ideas = []
    
    for i, strategy in enumerate(strategies_to_test):
        print(f"Testing {strategy.value} mutation:")
        
        # Select an idea to mutate
        idea_to_mutate = initial_ideas[i % len(initial_ideas)]
        
        try:
            # Apply mutation
            mutated_content = await mutation_manager.mutate(
                idea=idea_to_mutate,
                all_ideas=initial_ideas,
                generation=1,
                strategy=strategy,
                target_embedding=target_embedding
            )
            
            # Create mutated idea
            mutated_idea = Idea(
                id=f"mutated_{strategy.value}_{i}",
                content=mutated_content,
                generation=1,
                parent_ids=[idea_to_mutate.id],
                fitness=0.0  # Will be evaluated
            )
            
            # Add mock embedding and evaluate fitness
            mock_embedding = [0.15 + (i * 0.005) for j in range(384)]
            fitness_evaluator.add_embedding(mutated_idea.id, mock_embedding)
            fitness_evaluator.calculate_fitness(mutated_idea, initial_ideas + [mutated_idea], target_embedding)
            
            # Update mutation manager with fitness feedback
            mutation_manager.update_mutation_feedback(mutated_idea.id, mutated_idea.fitness)
            
            mutated_ideas.append(mutated_idea)
            
            print(f"  Original: {idea_to_mutate.content}")
            print(f"  Mutated:  {mutated_content}")
            print(f"  Fitness: {idea_to_mutate.fitness:.3f} -> {mutated_idea.fitness:.3f}")
            print(f"  Improvement: {mutated_idea.fitness - idea_to_mutate.fitness:+.3f}\n")
            
        except Exception as e:
            print(f"  Error applying {strategy.value} mutation: {e}\n")
    
    print("=== Mutation Performance Report ===\n")
    
    # Get performance report
    report = mutation_manager.get_performance_report()
    
    print("Strategy Performance:")
    for strategy, perf in report['strategy_performance'].items():
        if perf['usage_count'] > 0:
            print(f"  {strategy}:")
            print(f"    Success Rate: {perf['success_rate']:.3f}")
            print(f"    Avg Improvement: {perf['avg_improvement']:+.3f}")
            print(f"    Usage Count: {perf['usage_count']}")
    
    print(f"\nTotal Mutations: {report['total_mutations']}")
    print(f"Successful Mutations: {report['successful_mutations']}")
    print(f"Overall Success Rate: {report['successful_mutations'] / report['total_mutations']:.3f}")
    
    if report['component_rates']:
        print("\nComponent Mutation Rates:")
        for component, rates in report['component_rates'].items():
            print(f"  {component}:")
            print(f"    Current Rate: {rates['current_rate']:.3f}")
            print(f"    Success Rate: {rates['success_rate']:.3f}")
            print(f"    Adaptations: {rates['adaptations']}")
    
    print(f"\nCurrent Temperature (Simulated Annealing): {report['current_temperature']:.3f}")
    
    if report['successful_patterns']:
        print("\nSuccessful Patterns Learned:")
        for strategy, patterns in report['successful_patterns'].items():
            if patterns:
                print(f"  {strategy}: {len(patterns)} patterns")
                for pattern in patterns[:3]:  # Show first 3 patterns
                    print(f"    - {pattern}")
    
    print("\n=== Integration with Genetic Algorithm ===\n")
    
    # Create genetic algorithm with intelligent mutation enabled
    genetic_params = GeneticParameters(
        population_size=5,
        generations=2,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=1
    )
    
    genetic_algorithm = OptimizedGeneticAlgorithm(
        parameters=genetic_params,
        llm_client=llm_client,
        intelligent_mutation_enabled=True,
        target_embedding=target_embedding
    )
    
    print("Creating next generation with intelligent mutation enabled...")
    
    # Create next generation
    next_generation = await genetic_algorithm.create_next_generation_optimized(
        population=initial_ideas,
        generation=1,
        selection_method="tournament"
    )
    
    # Evaluate new generation
    for idea in next_generation:
        if idea.id not in fitness_evaluator.embeddings_cache:
            # Add mock embedding for new ideas
            mock_embedding = [0.12 + (len(idea.content) * 0.0001) for _ in range(384)]
            fitness_evaluator.add_embedding(idea.id, mock_embedding)
    
    fitness_evaluator.evaluate_population(next_generation, target_embedding)
    
    print("Next Generation:")
    for idea in next_generation:
        print(f"  {idea.id}: fitness={idea.fitness:.3f}")
        print(f"    Content: {idea.content[:100]}...")
        if len(idea.parent_ids) > 0:
            print(f"    Parents: {', '.join(idea.parent_ids)}")
        print()
    
    # Update mutation feedback for the new generation
    genetic_algorithm.update_mutation_feedback(next_generation)
    
    print("=== Final Performance Report ===\n")
    final_report = genetic_algorithm.get_mutation_performance_report()
    
    if final_report.get("message"):
        print(final_report["message"])
    else:
        print(f"Total Mutations Applied: {final_report.get('total_mutations', 0)}")
        print(f"Successful Mutations: {final_report.get('successful_mutations', 0)}")
        
        # Show improved strategies
        improved_strategies = []
        for strategy, perf in final_report.get('strategy_performance', {}).items():
            if perf['success_rate'] > 0.5 and perf['usage_count'] > 0:
                improved_strategies.append((strategy, perf['success_rate']))
        
        if improved_strategies:
            print("\nTop Performing Strategies:")
            for strategy, success_rate in sorted(improved_strategies, key=lambda x: x[1], reverse=True):
                print(f"  {strategy}: {success_rate:.3f} success rate")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    # Run the demonstration
    try:
        asyncio.run(demonstrate_intelligent_mutation())
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()