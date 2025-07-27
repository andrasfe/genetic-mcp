"""Example of using Claude evaluation to enhance genetic algorithm fitness.

This example demonstrates:
1. Creating a session with standard algorithmic evaluation
2. Enabling Claude evaluation mode
3. Having Claude evaluate ideas qualitatively
4. Combining algorithmic and Claude scores for better selection
"""

import asyncio
import json
from typing import Any, Dict
from fastmcp import FastMCP

# Initialize MCP client
mcp_client = FastMCP()


async def main():
    """Run genetic algorithm with Claude evaluation assistance."""
    
    print("=== Genetic Algorithm with Claude Evaluation ===\n")
    
    # Step 1: Create a session
    print("1. Creating session for idea generation...")
    session_params = {
        "prompt": "Ways to reduce food waste in restaurants",
        "mode": "iterative",
        "population_size": 20,
        "generations": 3,
        "top_k": 5
    }
    
    session = await mcp_client.call_tool("create_session", **session_params)
    session_id = session["session_id"]
    print(f"Created session: {session_id}")
    
    # Step 2: Enable Claude evaluation
    print("\n2. Enabling Claude evaluation mode...")
    await mcp_client.call_tool(
        "enable_claude_evaluation",
        session_id=session_id,
        evaluation_weight=0.4  # 40% Claude, 60% algorithmic
    )
    print("Claude evaluation enabled with 40% weight")
    
    # Step 3: Run first generation
    print("\n3. Running first generation...")
    result = await mcp_client.call_tool(
        "run_generation",
        session_id=session_id,
        top_k=10  # Get more ideas for evaluation
    )
    
    print(f"Generated {result['total_ideas_generated']} ideas")
    print(f"Top idea (algorithmic): {result['top_ideas'][0]['content'][:100]}...")
    
    # Step 4: Request Claude evaluation
    print("\n4. Requesting Claude to evaluate ideas...")
    eval_request = await mcp_client.call_tool(
        "evaluate_ideas",
        session_id=session_id,
        evaluation_batch_size=10
    )
    
    print(f"Ideas to evaluate: {eval_request['batch_size']}")
    print("\nEvaluation criteria:")
    for criterion, desc in eval_request['evaluation_instructions']['criteria'].items():
        print(f"  - {criterion}: {desc}")
    
    # Step 5: Simulate Claude's evaluation (in real usage, Claude would do this)
    print("\n5. Simulating Claude's qualitative evaluation...")
    evaluations = {}
    
    for idea in eval_request['ideas']:
        # Simulate thoughtful evaluation
        score = 0.7 + (hash(idea['content']) % 30) / 100  # 0.7-1.0 range
        
        evaluations[idea['id']] = {
            "score": score,
            "justification": f"This idea shows good potential for reducing food waste",
            "strengths": [
                "Practical implementation",
                "Measurable impact"
            ],
            "weaknesses": [
                "May require initial investment"
            ]
        }
    
    # Submit evaluations
    await mcp_client.call_tool(
        "submit_evaluations",
        session_id=session_id,
        evaluations=evaluations
    )
    print(f"Submitted {len(evaluations)} evaluations")
    
    # Step 6: Check updated rankings
    print("\n6. Checking updated idea rankings...")
    session_info = await mcp_client.call_tool(
        "get_session",
        session_id=session_id,
        include_ideas=True,
        ideas_limit=5
    )
    
    print(f"\nClaude evaluated: {session_info['claude_evaluated_count']} ideas")
    print("\nTop 5 ideas with combined fitness:")
    
    # Sort by combined fitness
    ideas = sorted(
        session_info['ideas'],
        key=lambda x: x.get('combined_fitness') or x['fitness'],
        reverse=True
    )[:5]
    
    for i, idea in enumerate(ideas, 1):
        print(f"\n{i}. {idea['content'][:80]}...")
        print(f"   Algorithmic fitness: {idea['fitness']:.3f}")
        if idea.get('claude_score') is not None:
            print(f"   Claude score: {idea['claude_score']:.3f}")
            print(f"   Combined fitness: {idea.get('combined_fitness', idea['fitness']):.3f}")
        
    # Step 7: Continue evolution with Claude-enhanced fitness
    print("\n\n7. Running next generation with Claude-enhanced selection...")
    result2 = await mcp_client.call_tool(
        "run_generation", 
        session_id=session_id,
        top_k=5
    )
    
    print(f"Generation 2 completed!")
    print("\nFinal top ideas:")
    for i, idea in enumerate(result2['top_ideas'][:3], 1):
        print(f"\n{i}. {idea['content']}")
        if idea.get('combined_fitness'):
            print(f"   Combined fitness: {idea['combined_fitness']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())