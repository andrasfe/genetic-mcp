"""Example demonstrating GPU-enhanced genetic algorithm with advanced features.

This example shows how to use the GPU-optimized genetic algorithm with:
- Advanced selection strategies (Boltzmann, SUS, rank-based)
- Fitness sharing and crowding distance
- Multi-population evolution
- Batch processing of multiple experiments
"""

import asyncio
import logging
import time
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from genetic_mcp.models import Idea
from genetic_mcp.genetic_algorithm_gpu_enhanced import (
    GPUEnhancedGeneticAlgorithm, AdvancedGeneticParameters
)
from genetic_mcp.gpu_batch_evolution import (
    GPUBatchEvolution, BatchEvolutionConfig, ExperimentConfig
)
from genetic_mcp.gpu_accelerated import GPUConfig
from genetic_mcp.llm_client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def compare_selection_strategies():
    """Compare different selection strategies on the same problem."""
    # Problem: Optimize a recommendation system
    target_prompt = "design an AI-powered recommendation system for e-commerce"
    
    # Selection strategies to compare
    strategies = ["tournament", "boltzmann", "sus", "rank", "diversity"]
    results = {}
    
    # GPU configuration
    gpu_config = GPUConfig(
        device="cuda",  # Use "cpu" if no GPU available
        batch_size=64,
        use_mixed_precision=True
    )
    
    # Initialize LLM client (optional)
    llm_client = None  # LLMClient() if you have API keys configured
    
    for strategy in strategies:
        logger.info(f"\n=== Testing {strategy} selection ===")
        
        # Configure parameters
        params = AdvancedGeneticParameters(
            population_size=50,
            generations=20,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_count=5,
            selection_method=strategy,
            tournament_size=3,
            selection_pressure=0.9,
            temperature_initial=1.0,
            temperature_decay=0.95,
            use_fitness_sharing=strategy == "diversity",
            sigma_share=0.1,
            diversity_weight=0.3
        )
        
        # Initialize algorithm
        ga = GPUEnhancedGeneticAlgorithm(
            parameters=params,
            llm_client=llm_client,
            gpu_config=gpu_config
        )
        
        # Create initial population
        initial_population = create_initial_population(params.population_size)
        
        # Track evolution progress
        evolution_data = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
        
        async def on_generation(gen, population, diversity_metrics):
            """Callback to track progress."""
            fitness_values = [idea.fitness for idea in population]
            evolution_data['best_fitness'].append(max(fitness_values))
            evolution_data['avg_fitness'].append(np.mean(fitness_values))
            evolution_data['diversity'].append(diversity_metrics.get('embedding_diversity', 0))
        
        # Run evolution
        start_time = time.time()
        
        final_population = await ga.evolve_population(
            initial_population,
            target_prompt,
            params.generations,
            callbacks={'on_generation': on_generation}
        )
        
        elapsed_time = time.time() - start_time
        
        # Get summary
        summary = ga.get_evolution_summary()
        summary['evolution_data'] = evolution_data
        summary['elapsed_time'] = elapsed_time
        
        results[strategy] = summary
        
        # Clean up
        ga.cleanup()
        
        # Display top solutions
        logger.info(f"\nTop 3 solutions for {strategy}:")
        top_ideas = sorted(final_population, key=lambda x: x.fitness, reverse=True)[:3]
        for i, idea in enumerate(top_ideas):
            logger.info(f"{i+1}. (Fitness: {idea.fitness:.3f}) {idea.content[:100]}...")
    
    # Plot comparison
    plot_strategy_comparison(results)
    
    return results


async def multi_population_evolution():
    """Demonstrate multi-population evolution with migration."""
    logger.info("\n=== Multi-Population Evolution Demo ===")
    
    target_prompt = "develop a sustainable smart city infrastructure"
    
    # Configure multi-population parameters
    params = AdvancedGeneticParameters(
        population_size=100,
        generations=30,
        mutation_rate=0.15,
        crossover_rate=0.85,
        elitism_count=10,
        selection_method="tournament",
        n_subpopulations=4,  # 4 subpopulations
        migration_rate=0.1,   # 10% migration
        migration_interval=5,  # Every 5 generations
        adaptive_mutation=True,
        use_crowding=True
    )
    
    gpu_config = GPUConfig(device="cuda", batch_size=128)
    
    # Initialize algorithm
    ga = GPUEnhancedGeneticAlgorithm(
        parameters=params,
        gpu_config=gpu_config
    )
    
    # Create initial population
    initial_population = create_diverse_initial_population(params.population_size)
    
    # Track subpopulation statistics
    subpop_stats = []
    
    async def on_generation(gen, population, diversity_metrics):
        """Track subpopulation diversity."""
        stats = {
            'generation': gen,
            'overall_diversity': diversity_metrics.get('embedding_diversity', 0),
            'cluster_entropy': diversity_metrics.get('cluster_entropy', 0),
            'n_effective_clusters': diversity_metrics.get('n_effective_clusters', 0)
        }
        subpop_stats.append(stats)
        
        if gen % 5 == 0:
            logger.info(f"Gen {gen}: Diversity={stats['overall_diversity']:.3f}, "
                       f"Clusters={stats['n_effective_clusters']}")
    
    # Run evolution
    start_time = time.time()
    
    final_population = await ga.evolve_population(
        initial_population,
        target_prompt,
        params.generations,
        callbacks={'on_generation': on_generation}
    )
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"\nEvolution completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final best fitness: {max(idea.fitness for idea in final_population):.3f}")
    
    # Analyze population structure
    analyze_population_structure(final_population, subpop_stats)
    
    ga.cleanup()
    
    return final_population, subpop_stats


async def batch_experiment_processing():
    """Demonstrate batch processing of multiple experiments."""
    logger.info("\n=== Batch Experiment Processing Demo ===")
    
    # Configure batch processing
    batch_config = BatchEvolutionConfig(
        n_experiments=10,
        max_batch_size=200,
        async_workers=4,
        checkpoint_interval=10
    )
    
    gpu_config = GPUConfig(
        device="cuda",
        batch_size=256,
        memory_fraction=0.9
    )
    
    # Create experiment configurations
    experiment_configs = []
    prompts = [
        "optimize supply chain logistics",
        "design personalized learning system",
        "create sustainable energy grid",
        "develop healthcare AI assistant",
        "build autonomous transportation network"
    ]
    
    # Create 2 experiments for each prompt with different parameters
    for i, prompt in enumerate(prompts):
        for variant in range(2):
            config = ExperimentConfig(
                experiment_id=f"exp_{i}_{variant}",
                population_size=30,
                target_prompt=prompt,
                parameters=AdvancedGeneticParameters(
                    population_size=30,
                    mutation_rate=0.1 + variant * 0.1,
                    crossover_rate=0.7 + variant * 0.1,
                    selection_method="tournament" if variant == 0 else "rank"
                ),
                metadata={'prompt_category': i, 'variant': variant}
            )
            experiment_configs.append(config)
    
    # Initialize batch processor
    batch_processor = GPUBatchEvolution(batch_config, gpu_config)
    
    # Track batch progress
    batch_metrics = []
    
    async def on_batch_generation(gen, experiments):
        """Track batch processing metrics."""
        metrics = {
            'generation': gen,
            'experiments': {}
        }
        
        for exp_id, state in experiments.items():
            if state.fitness_history:
                metrics['experiments'][exp_id] = {
                    'best_fitness': max(state.fitness_history[-1]) if state.fitness_history[-1] else 0,
                    'diversity': state.diversity_history[-1].get('embedding_diversity', 0) 
                               if state.diversity_history else 0
                }
        
        batch_metrics.append(metrics)
        
        if gen % 5 == 0:
            avg_fitness = np.mean([m['best_fitness'] for m in metrics['experiments'].values()])
            logger.info(f"Batch Gen {gen}: Avg best fitness across experiments: {avg_fitness:.3f}")
    
    # Run batch experiments
    start_time = time.time()
    
    results = await batch_processor.run_batch_experiments(
        experiment_configs,
        generations=20,
        callbacks={'on_batch_generation': on_batch_generation}
    )
    
    elapsed_time = time.time() - start_time
    
    # Get summary
    summary = batch_processor.get_batch_summary()
    
    logger.info(f"\nBatch processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {len(results)} experiments")
    logger.info(f"Average time per experiment: {elapsed_time / len(results):.2f} seconds")
    
    # Analyze batch results
    analyze_batch_results(results, batch_metrics, experiment_configs)
    
    return results, batch_metrics


def create_initial_population(size: int) -> List[Idea]:
    """Create initial population with diverse ideas."""
    templates = [
        "Implement {} using deep learning and neural networks",
        "Create a distributed {} with microservices architecture",
        "Design {} leveraging blockchain technology",
        "Build {} with real-time data processing capabilities",
        "Develop {} using edge computing and IoT sensors"
    ]
    
    population = []
    for i in range(size):
        template = templates[i % len(templates)]
        content = template.format("the solution") + f" - variant {i}"
        idea = Idea(
            id=f"initial_{i}",
            content=content,
            generation=0
        )
        population.append(idea)
    
    return population


def create_diverse_initial_population(size: int) -> List[Idea]:
    """Create initial population with intentionally diverse ideas."""
    categories = [
        ("technology", ["AI/ML", "blockchain", "quantum", "IoT", "cloud"]),
        ("approach", ["centralized", "distributed", "hybrid", "modular", "monolithic"]),
        ("scale", ["local", "regional", "national", "global", "universal"]),
        ("focus", ["efficiency", "sustainability", "security", "accessibility", "innovation"])
    ]
    
    population = []
    for i in range(size):
        # Randomly combine elements from different categories
        elements = []
        for _, options in categories:
            elements.append(np.random.choice(options))
        
        content = f"Develop solution using {elements[0]} with {elements[1]} architecture at {elements[2]} scale focusing on {elements[3]}"
        
        idea = Idea(
            id=f"diverse_{i}",
            content=content,
            generation=0
        )
        population.append(idea)
    
    return population


def plot_strategy_comparison(results: Dict[str, Dict]):
    """Plot comparison of different selection strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Selection Strategy Comparison', fontsize=16)
    
    strategies = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    # Best fitness over generations
    ax = axes[0, 0]
    for strategy, color in zip(strategies, colors):
        data = results[strategy]['evolution_data']
        ax.plot(data['best_fitness'], label=strategy, color=color, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best Fitness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average fitness over generations
    ax = axes[0, 1]
    for strategy, color in zip(strategies, colors):
        data = results[strategy]['evolution_data']
        ax.plot(data['avg_fitness'], label=strategy, color=color, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Average Fitness Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Diversity over generations
    ax = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        data = results[strategy]['evolution_data']
        if data['diversity']:
            ax.plot(data['diversity'], label=strategy, color=color, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Diversity')
    ax.set_title('Diversity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance comparison
    ax = axes[1, 1]
    final_fitness = [results[s]['final_best_fitness'] for s in strategies]
    elapsed_times = [results[s]['elapsed_time'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_fitness, width, label='Final Fitness', color='skyblue')
    bars2 = ax2.bar(x + width/2, elapsed_times, width, label='Time (s)', color='lightcoral')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Final Best Fitness', color='skyblue')
    ax2.set_ylabel('Elapsed Time (s)', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45)
    ax.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    
    plt.tight_layout()
    plt.savefig('selection_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_population_structure(population: List[Idea], subpop_stats: List[Dict]):
    """Analyze and visualize population structure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Multi-Population Evolution Analysis', fontsize=16)
    
    # Diversity metrics over time
    ax = axes[0]
    generations = [s['generation'] for s in subpop_stats]
    diversity = [s['overall_diversity'] for s in subpop_stats]
    clusters = [s['n_effective_clusters'] for s in subpop_stats]
    
    ax.plot(generations, diversity, 'b-', label='Population Diversity', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(generations, clusters, 'r--', label='Effective Clusters', linewidth=2)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity', color='b')
    ax2.set_ylabel('Number of Clusters', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Final population fitness distribution
    ax = axes[1]
    fitness_values = [idea.fitness for idea in population]
    ax.hist(fitness_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(fitness_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(fitness_values):.3f}')
    ax.axvline(np.max(fitness_values), color='blue', linestyle='--',
               label=f'Max: {np.max(fitness_values):.3f}')
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Count')
    ax.set_title('Final Population Fitness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_population_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_batch_results(results: Dict, batch_metrics: List[Dict], configs: List[ExperimentConfig]):
    """Analyze and visualize batch processing results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch Evolution Analysis', fontsize=16)
    
    # Performance by prompt category
    ax = axes[0, 0]
    prompt_categories = {}
    for config in configs:
        cat = config.metadata['prompt_category']
        if cat not in prompt_categories:
            prompt_categories[cat] = []
        
        final_pop = results[config.experiment_id]
        best_fitness = max(idea.fitness for idea in final_pop)
        prompt_categories[cat].append(best_fitness)
    
    categories = sorted(prompt_categories.keys())
    cat_means = [np.mean(prompt_categories[c]) for c in categories]
    cat_stds = [np.std(prompt_categories[c]) for c in categories]
    
    x = np.arange(len(categories))
    ax.bar(x, cat_means, yerr=cat_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Prompt Category')
    ax.set_ylabel('Final Best Fitness')
    ax.set_title('Performance by Problem Category')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cat {c}' for c in categories])
    ax.grid(True, alpha=0.3)
    
    # Fitness progression over generations
    ax = axes[0, 1]
    for i, metric in enumerate(batch_metrics[::2]):  # Sample every other generation
        gen = metric['generation']
        fitness_values = [m['best_fitness'] for m in metric['experiments'].values()]
        positions = np.random.normal(gen, 0.1, len(fitness_values))
        ax.scatter(positions, fitness_values, alpha=0.5, s=20)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Fitness Distribution Across Experiments')
    ax.grid(True, alpha=0.3)
    
    # Selection method comparison
    ax = axes[1, 0]
    tournament_results = []
    rank_results = []
    
    for config in configs:
        final_pop = results[config.experiment_id]
        best_fitness = max(idea.fitness for idea in final_pop)
        
        if config.parameters.selection_method == "tournament":
            tournament_results.append(best_fitness)
        else:
            rank_results.append(best_fitness)
    
    data = [tournament_results, rank_results]
    labels = ['Tournament', 'Rank']
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Final Best Fitness')
    ax.set_title('Selection Method Comparison')
    ax.grid(True, alpha=0.3)
    
    # GPU utilization estimate
    ax = axes[1, 1]
    n_ideas_per_gen = sum(config.population_size for config in configs)
    generations = len(batch_metrics)
    total_evaluations = n_ideas_per_gen * generations
    
    info_text = f"""Batch Processing Statistics:
    
Total Experiments: {len(configs)}
Total Evaluations: {total_evaluations:,}
Ideas per Generation: {n_ideas_per_gen}
Generations: {generations}

Average Final Fitness: {np.mean([max(idea.fitness for idea in pop) for pop in results.values()]):.3f}
Best Overall Fitness: {max(max(idea.fitness for idea in pop) for pop in results.values()):.3f}
    """
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_evolution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


async def main():
    """Run all examples."""
    logger.info("Starting GPU-Enhanced Genetic Algorithm Examples")
    
    # Example 1: Compare selection strategies
    logger.info("\n" + "="*50)
    logger.info("Example 1: Comparing Selection Strategies")
    logger.info("="*50)
    
    try:
        strategy_results = await compare_selection_strategies()
        logger.info("\nStrategy comparison completed successfully!")
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
    
    # Example 2: Multi-population evolution
    logger.info("\n" + "="*50)
    logger.info("Example 2: Multi-Population Evolution")
    logger.info("="*50)
    
    try:
        multi_pop_results = await multi_population_evolution()
        logger.info("\nMulti-population evolution completed successfully!")
    except Exception as e:
        logger.error(f"Multi-population evolution failed: {e}")
    
    # Example 3: Batch experiment processing
    logger.info("\n" + "="*50)
    logger.info("Example 3: Batch Experiment Processing")
    logger.info("="*50)
    
    try:
        batch_results = await batch_experiment_processing()
        logger.info("\nBatch processing completed successfully!")
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("All examples completed!")
    logger.info("Check the generated PNG files for visualizations.")
    logger.info("="*50)


if __name__ == "__main__":
    asyncio.run(main())