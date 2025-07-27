#!/usr/bin/env python3
"""Example demonstrating client-generated ideas functionality in the Genetic MCP server."""

import asyncio
import json
import os
import sys
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_mcp import server
from genetic_mcp.models import FitnessWeights, GenerationRequest, EvolutionMode
from genetic_mcp.session_manager import SessionManager
from genetic_mcp.llm_client import MultiModelClient, OpenRouterClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class ClientIdeaGenerator:
    """Example client that generates ideas for the genetic algorithm."""
    
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.generation_templates = {
            0: [  # Initial generation - diverse approaches
                "A solution focusing on {aspect} for {prompt}",
                "An innovative approach using {tech} to address {prompt}",
                "A sustainable method that {action} to solve {prompt}",
                "A cost-effective system that {mechanism} for {prompt}",
                "A user-centered design that {feature} to handle {prompt}"
            ],
            1: [  # Generation 1 - combine and refine
                "Hybrid system combining {parent1} with {parent2}",
                "Enhanced version of {parent} with added {improvement}",
                "Integrated platform that merges {aspect1} and {aspect2}",
                "Advanced implementation of {parent} using {technology}",
                "Optimized solution based on {parent} with {efficiency}"
            ],
            2: [  # Generation 2 - further evolution
                "Next-generation {concept} with AI-powered {feature}",
                "Fully integrated {system} combining best aspects of previous ideas",
                "Revolutionary approach merging {tech1} and {tech2}",
                "Ultimate solution incorporating {feature1}, {feature2}, and {feature3}",
                "Future-ready platform with {capability1} and {capability2}"
            ]
        }
    
    def generate_ideas_for_generation(self, generation: int, 
                                    parent_ideas: List[str] = None) -> List[str]:
        """Generate ideas for a specific generation."""
        ideas = []
        templates = self.generation_templates.get(generation, self.generation_templates[0])
        
        if generation == 0:
            # Initial generation - create diverse ideas
            aspects = ["technology", "efficiency", "sustainability", "accessibility", "scalability"]
            techs = ["AI", "blockchain", "IoT", "quantum computing", "biotechnology"]
            actions = ["optimizes", "revolutionizes", "transforms", "enhances", "streamlines"]
            mechanisms = ["leverages data", "uses automation", "applies ML", "integrates systems", "coordinates resources"]
            features = ["provides real-time updates", "ensures security", "maximizes efficiency", "minimizes waste", "enhances UX"]
            
            for i, template in enumerate(templates):
                idea = template.format(
                    aspect=aspects[i % len(aspects)],
                    tech=techs[i % len(techs)],
                    action=actions[i % len(actions)],
                    mechanism=mechanisms[i % len(mechanisms)],
                    feature=features[i % len(features)],
                    prompt=self.prompt
                )
                ideas.append(idea)
        
        elif parent_ideas:
            # Later generations - evolve from parents
            improvements = ["real-time analytics", "predictive capabilities", "automated optimization", "smart integration", "adaptive learning"]
            technologies = ["machine learning", "edge computing", "5G networks", "renewable energy", "nanotechnology"]
            efficiencies = ["90% reduction in costs", "10x performance boost", "zero-emission operation", "instant response times", "global scalability"]
            
            for i, template in enumerate(templates):
                if i < len(parent_ideas):
                    parent = parent_ideas[i % len(parent_ideas)].split('.')[0]  # Get core idea
                    
                    if "{parent1}" in template and "{parent2}" in template and len(parent_ideas) > 1:
                        idea = template.format(
                            parent1=parent_ideas[0].split('.')[0],
                            parent2=parent_ideas[1].split('.')[0]
                        )
                    elif "{parent}" in template:
                        idea = template.format(
                            parent=parent,
                            improvement=improvements[i % len(improvements)],
                            technology=technologies[i % len(technologies)],
                            efficiency=efficiencies[i % len(efficiencies)]
                        )
                    else:
                        # Generic evolution template
                        idea = f"Advanced evolution of '{parent}' with {improvements[i % len(improvements)]}"
                    
                    ideas.append(idea)
        
        return ideas[:5]  # Return exactly 5 ideas


async def demonstrate_client_generated_mode():
    """Demonstrate the client-generated ideas functionality."""
    
    print("=== Genetic MCP Client-Generated Ideas Demo ===\n")
    
    # Initialize the server components
    llm_client = server.initialize_llm_client()
    session_manager = SessionManager(llm_client)
    await session_manager.start()
    
    # Set the global session manager
    server.session_manager = session_manager
    
    try:
        # Define the problem
        prompt = "innovative solutions for reducing plastic waste in oceans"
        print(f"Problem: {prompt}\n")
        
        # Create a client-generated session
        session_info = await server.create_session(
            prompt=prompt,
            mode="iterative",
            population_size=5,
            top_k=3,
            generations=3,
            fitness_weights={"relevance": 0.4, "novelty": 0.3, "feasibility": 0.3},
            client_generated=True
        )
        
        session_id = session_info["session_id"]
        print(f"Created session: {session_id}")
        print(f"Mode: {session_info['mode']}")
        print(f"Client-generated: {session_info['client_generated']}")
        print(f"Population size: {session_info['parameters']['population_size']}")
        print(f"Generations: {session_info['parameters']['generations']}\n")
        
        # Initialize our client idea generator
        idea_generator = ClientIdeaGenerator(prompt)
        
        # Start the genetic algorithm in a background task
        print("Starting genetic algorithm...")
        generation_task = asyncio.create_task(
            server.run_generation(session_id, top_k=3)
        )
        
        # Let it start and wait for ideas
        await asyncio.sleep(0.5)
        
        # Generate and inject ideas for each generation
        parent_ideas = None
        for gen in range(3):
            print(f"\n--- Generation {gen} ---")
            
            # Generate ideas
            ideas = idea_generator.generate_ideas_for_generation(gen, parent_ideas)
            print(f"Generated {len(ideas)} ideas:")
            for i, idea in enumerate(ideas, 1):
                print(f"  {i}. {idea}")
            
            # Inject ideas into the session
            injection_result = await server.inject_ideas(
                session_id=session_id,
                ideas=ideas,
                generation=gen
            )
            
            print(f"Injected {injection_result['injected_count']} ideas")
            
            # Store best ideas as parents for next generation
            if gen < 2:  # Don't need parents for after last generation
                await asyncio.sleep(1)  # Give time for fitness evaluation
                session_data = await server.get_session(session_id, include_ideas=True, generation_filter=gen)
                if session_data.get("ideas"):
                    # Get top ideas from this generation
                    gen_ideas = sorted(session_data["ideas"], key=lambda x: x["fitness"], reverse=True)
                    parent_ideas = [idea["content"] for idea in gen_ideas[:2]]  # Top 2 as parents
        
        # Wait for the algorithm to complete
        print("\nWaiting for genetic algorithm to complete...")
        result = await generation_task
        
        # Display results
        print(f"\n=== Results ===")
        print(f"Total ideas generated: {result['total_ideas_generated']}")
        print(f"Generations completed: {result['generations_completed']}")
        print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
        
        print(f"\nTop {len(result['top_ideas'])} ideas:")
        for i, idea in enumerate(result['top_ideas'], 1):
            print(f"\n{i}. {idea['content']}")
            print(f"   Fitness: {idea['fitness']:.3f}")
            print(f"   Generation: {idea['generation']}")
            scores = idea['scores']
            print(f"   Scores: Relevance={scores.get('relevance', 0):.3f}, "
                  f"Novelty={scores.get('novelty', 0):.3f}, "
                  f"Feasibility={scores.get('feasibility', 0):.3f}")
            
            # Show lineage if available
            if idea.get('parent_ids'):
                print(f"   Parents: {', '.join(idea['parent_ids'][:2])}")  # Show first 2 parent IDs
        
        # Get final session state
        final_session = await server.get_session(session_id, include_ideas=False)
        print(f"\nIdeas per generation: {final_session['ideas_per_generation']}")
        
    finally:
        await session_manager.stop()
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(demonstrate_client_generated_mode())