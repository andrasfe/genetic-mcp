#!/usr/bin/env python3
"""
Example demonstrating advanced crossover operators in genetic-mcp.

This example shows how to:
1. Configure advanced crossover operators
2. Use different crossover strategies
3. Enable adaptive crossover selection
4. Track crossover performance
"""

import asyncio
import json
import os
from genetic_mcp.models import GenerationRequest, EvolutionMode
from genetic_mcp.advanced_crossover import CrossoverOperator


async def example_advanced_crossover():
    """Demonstrate advanced crossover operators."""
    
    print("Advanced Crossover Operators Example")
    print("=" * 50)
    
    # Example 1: Using a specific crossover operator
    print("\n1. Using Multi-Point Crossover")
    print("-" * 30)
    
    request = GenerationRequest(
        prompt="Develop innovative solutions for sustainable urban transportation",
        mode=EvolutionMode.ITERATIVE,
        population_size=8,
        generations=3,
        top_k=3,
        advanced_crossover_enabled=True,
        crossover_strategy="multi_point",  # Specify operator
        crossover_config={
            "num_points": 2  # Custom configuration for multi-point
        }
    )
    
    # Create session (this would normally connect to the MCP server)
    session_data = {
        "prompt": request.prompt,
        "mode": request.mode.value,
        "population_size": request.population_size,
        "generations": request.generations,
        "top_k": request.top_k,
        "advanced_crossover_enabled": request.advanced_crossover_enabled,
        "crossover_strategy": request.crossover_strategy,
        "crossover_config": request.crossover_config or {}
    }
    
    print(f"Session configured with:")
    print(f"  - Crossover Strategy: {session_data['crossover_strategy']}")
    print(f"  - Crossover Config: {session_data['crossover_config']}")
    
    # Example 2: Using adaptive crossover selection
    print("\n2. Using Adaptive Crossover Selection")
    print("-" * 40)
    
    adaptive_request = GenerationRequest(
        prompt="Create AI-powered educational tools for remote learning",
        mode=EvolutionMode.ITERATIVE,
        population_size=10,
        generations=4,
        top_k=5,
        advanced_crossover_enabled=True,
        crossover_strategy=None,  # Let the system choose adaptively
        crossover_adaptation_enabled=True
    )
    
    adaptive_session = {
        "prompt": adaptive_request.prompt,
        "advanced_crossover_enabled": True,
        "crossover_strategy": None,  # Adaptive selection
        "crossover_adaptation_enabled": True
    }
    
    print(f"Adaptive session configured:")
    print(f"  - Adaptive Selection: {adaptive_session['crossover_adaptation_enabled']}")
    
    # Example 3: Available crossover operators
    print("\n3. Available Crossover Operators")
    print("-" * 35)
    
    operators = {
        CrossoverOperator.SEMANTIC: "LLM-guided semantic blending",
        CrossoverOperator.MULTI_POINT: "Split ideas at multiple points",
        CrossoverOperator.UNIFORM: "Use learned masks for selection",
        CrossoverOperator.EDGE_RECOMBINATION: "Preserve concept relationships",
        CrossoverOperator.ORDER_BASED: "Maintain logical flow",
        CrossoverOperator.BLEND: "Fitness-weighted combination",
        CrossoverOperator.CONCEPT_MAPPING: "Map and recombine concepts",
        CrossoverOperator.SYNTACTIC: "Grammar-aware crossover",
        CrossoverOperator.HIERARCHICAL: "Tree-based structure crossover",
        CrossoverOperator.ADAPTIVE: "Dynamically choose best operator"
    }
    
    for operator, description in operators.items():
        print(f"  - {operator.value}: {description}")
    
    # Example 4: Configuration options
    print("\n4. Crossover Configuration Options")
    print("-" * 37)
    
    config_examples = {
        "multi_point": {
            "num_points": 2,  # Number of crossover points
            "mask_probability": 0.5  # Probability for uniform crossover
        },
        "blend": {
            "alpha": 0.3,  # Blending parameter
            "fitness_weighted": True  # Use fitness-based weighting
        },
        "edge_recombination": {
            "preserve_order": True,  # Maintain concept ordering
            "concept_threshold": 0.7  # Similarity threshold for concepts
        },
        "adaptive": {
            "exploration_factor": 0.1,  # UCB exploration parameter
            "min_usage_threshold": 5  # Min uses before performance-based selection
        }
    }
    
    for operator, config in config_examples.items():
        print(f"  {operator}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    # Example 5: Performance tracking
    print("\n5. Performance Tracking")
    print("-" * 25)
    
    print("Advanced crossover operators track performance metrics:")
    print("  - Usage count per operator")
    print("  - Success rate per operator")
    print("  - Average fitness improvement")
    print("  - Parent similarity scores")
    print("  - Offspring diversity scores")
    
    print("\nExample performance report structure:")
    example_report = {
        "total_crossovers": 25,
        "operator_performance": {
            "semantic": {
                "usage_count": 8,
                "success_rate": 1.0,
                "avg_fitness_improvement": 0.15
            },
            "multi_point": {
                "usage_count": 6,
                "success_rate": 0.83,
                "avg_fitness_improvement": 0.12
            },
            "adaptive": {
                "usage_count": 11,
                "success_rate": 0.91,
                "avg_fitness_improvement": 0.18
            }
        },
        "recommended_operator": "adaptive"
    }
    
    print(json.dumps(example_report, indent=2))
    
    # Example 6: Integration with MCP server
    print("\n6. MCP Server Integration")
    print("-" * 28)
    
    print("To use advanced crossover with the MCP server:")
    print("""
    # Client-side request:
    {
      "method": "create_session",
      "params": {
        "prompt": "Your prompt here",
        "mode": "iterative",
        "population_size": 10,
        "generations": 5,
        "advanced_crossover_enabled": true,
        "crossover_strategy": "adaptive",  // or specific operator
        "crossover_config": {
          "exploration_factor": 0.15
        }
      }
    }
    """)
    
    print("\nThe server will:")
    print("  1. Initialize AdvancedCrossoverManager")
    print("  2. Use specified or adaptive crossover operators")
    print("  3. Track performance across generations")
    print("  4. Provide performance reports via get_optimization_report")
    
    print("\n" + "=" * 50)
    print("Example completed! Check the documentation for more details.")


if __name__ == "__main__":
    asyncio.run(example_advanced_crossover())