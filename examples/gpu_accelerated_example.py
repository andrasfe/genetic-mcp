#!/usr/bin/env python3
"""Example of using GPU-accelerated genetic algorithm in the MCP server."""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from genetic_mcp.models import (
    Session, Idea, FitnessWeights, GeneticParameters,
    GenerationProgress, EvolutionMode
)
from genetic_mcp.gpu_accelerated import GPUConfig
from genetic_mcp.gpu_integration import (
    GPUAcceleratedSessionManager,
    GPUBatchIdeaProcessor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_gpu_acceleration():
    """Demonstrate GPU-accelerated genetic algorithm."""
    
    # 1. Configure GPU settings
    gpu_config = GPUConfig(
        device="cuda",  # Use "cpu" if no GPU available
        batch_size=64,
        max_sequence_length=512,
        use_mixed_precision=True,
        memory_fraction=0.8
    )
    
    logger.info(f"Using device: {gpu_config.device}")
    
    # 2. Create GPU-accelerated session manager
    session_manager = GPUAcceleratedSessionManager(gpu_config=gpu_config)
    
    # 3. Warm up GPU (optional but recommended)
    logger.info("Warming up GPU...")
    await session_manager.warm_up_gpu()
    
    # 4. Create a sample session
    session = Session(
        id="demo_session",
        client_id="demo_client",
        prompt="Generate innovative ideas for reducing carbon emissions in urban transportation",
        mode=EvolutionMode.ITERATIVE,
        parameters=GeneticParameters(
            population_size=50,
            generations=5,
            mutation_rate=0.15,
            crossover_rate=0.7,
            elitism_count=5
        ),
        fitness_weights=FitnessWeights(
            relevance=0.4,
            novelty=0.3,
            feasibility=0.3
        )
    )
    
    # 5. Create initial population
    initial_ideas = [
        Idea(
            id=f"idea_{i}",
            content=generate_sample_idea(i),
            generation=0
        )
        for i in range(session.parameters.population_size)
    ]
    session.ideas = initial_ideas
    
    logger.info(f"Starting evolution with {len(initial_ideas)} initial ideas")
    
    # 6. Run genetic algorithm evolution
    all_fitness_scores = []
    
    for generation in range(session.parameters.generations):
        logger.info(f"\n--- Generation {generation + 1} ---")
        
        # Get current population
        start_idx = max(0, len(session.ideas) - session.parameters.population_size)
        current_population = session.ideas[start_idx:]
        
        # Process generation with GPU acceleration
        new_ideas, progress = await session_manager.process_generation_gpu(
            session,
            current_population,
            session.prompt
        )
        
        # Log progress
        logger.info(f"Best fitness: {progress.best_fitness:.3f}")
        logger.info(f"Total ideas generated: {len(session.ideas)}")
        logger.info(f"Active workers: {progress.active_workers}")
        
        # Collect fitness scores
        all_fitness_scores.append(progress.best_fitness)
        
        # Show GPU memory stats
        if generation % 2 == 0:
            stats = session_manager.get_gpu_memory_stats()
            logger.info(f"GPU Memory - Allocated: {stats['gpu_memory']['allocated']:.2f}GB, "
                       f"Free: {stats['gpu_memory']['free']:.2f}GB")
    
    # 7. Get final results with diversity
    logger.info("\n--- Final Results ---")
    
    # Find diverse top ideas
    top_ideas = await session_manager.find_optimal_population_subset(
        session,
        k=10,
        diversity_weight=0.5  # Balance between fitness and diversity
    )
    
    # Display top ideas
    for i, idea in enumerate(top_ideas[:5], 1):
        logger.info(f"\nTop Idea {i} (fitness: {idea.fitness:.3f}):")
        logger.info(f"Content: {idea.content[:200]}...")
        logger.info(f"Scores - Relevance: {idea.scores.get('relevance', 0):.3f}, "
                   f"Novelty: {idea.scores.get('novelty', 0):.3f}, "
                   f"Feasibility: {idea.scores.get('feasibility', 0):.3f}")
    
    # 8. Show evolution progress
    logger.info("\n--- Evolution Progress ---")
    for i, fitness in enumerate(all_fitness_scores):
        logger.info(f"Generation {i + 1}: Best fitness = {fitness:.3f}")
    
    # 9. Clean up GPU memory
    session_manager.clear_gpu_caches()
    logger.info("\nGPU caches cleared")
    
    return top_ideas


async def demonstrate_batch_processing():
    """Demonstrate batch processing multiple sessions."""
    
    logger.info("\n=== Batch Processing Demo ===")
    
    # Configure GPU
    gpu_config = GPUConfig(
        device="cuda",
        batch_size=128,  # Larger batch for multiple sessions
        use_mixed_precision=True
    )
    
    # Create batch processor
    batch_processor = GPUBatchIdeaProcessor(gpu_config=gpu_config)
    await batch_processor.start()
    
    # Create multiple sessions with different prompts
    prompts = [
        "Innovative solutions for renewable energy storage",
        "Ideas for sustainable urban farming",
        "Concepts for reducing plastic waste in oceans"
    ]
    
    sessions = []
    for i, prompt in enumerate(prompts):
        session = Session(
            id=f"batch_session_{i}",
            client_id=f"client_{i}",
            prompt=prompt,
            mode=EvolutionMode.SINGLE_PASS,
            parameters=GeneticParameters(population_size=30),
            fitness_weights=FitnessWeights()
        )
        
        # Add initial ideas
        session.ideas = [
            Idea(
                id=f"session{i}_idea{j}",
                content=f"Initial idea {j} for {prompt}",
                generation=0
            )
            for j in range(30)
        ]
        sessions.append(session)
    
    # Process all sessions in parallel
    logger.info(f"Processing {len(sessions)} sessions in parallel...")
    
    futures = []
    for session in sessions:
        future = await batch_processor.add_to_batch(
            session.id,
            session.ideas,
            session.prompt,
            {
                "relevance": session.fitness_weights.relevance,
                "novelty": session.fitness_weights.novelty,
                "feasibility": session.fitness_weights.feasibility
            }
        )
        futures.append(future)
    
    # Wait for results
    results = await asyncio.gather(*futures)
    
    # Display results
    for i, (session, result) in enumerate(zip(sessions, results)):
        logger.info(f"\nSession {i} ({session.prompt[:50]}...):")
        logger.info(f"Average fitness: {result['fitness_scores'].mean():.3f}")
        logger.info(f"Best fitness: {result['fitness_scores'].max():.3f}")
    
    # Stop batch processor
    await batch_processor.stop()
    
    logger.info("\nBatch processing completed")


def generate_sample_idea(index: int) -> str:
    """Generate sample transportation idea content."""
    ideas = [
        "Implement AI-powered traffic light optimization to reduce idle time and emissions",
        "Develop electric autonomous shuttle networks for last-mile connectivity",
        "Create dynamic carpooling lanes that adapt to real-time traffic patterns",
        "Deploy solar-powered charging stations integrated with public transit stops",
        "Design modular electric bikes with swappable battery systems",
        "Introduce gamified public transit apps that reward eco-friendly travel",
        "Build underground automated parking systems to reduce surface congestion",
        "Implement drone delivery networks to reduce delivery truck emissions",
        "Create green corridors with dedicated lanes for zero-emission vehicles",
        "Develop hydrogen fuel cell buses with regenerative braking systems"
    ]
    
    base_idea = ideas[index % len(ideas)]
    variation = f" Enhanced with feature variant {index // len(ideas) + 1}."
    
    return base_idea + variation


async def main():
    """Run all demonstrations."""
    try:
        # Check if GPU is available
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("No GPU detected, will use CPU")
        except ImportError:
            logger.warning("PyTorch not installed, GPU acceleration unavailable")
            return
        
        # Run demonstrations
        await demonstrate_gpu_acceleration()
        await demonstrate_batch_processing()
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())