#!/usr/bin/env python3
"""Example demonstrating the genetic-mcp persistence system.

This example shows how to:
1. Create and save sessions
2. Load and resume sessions
3. Work with checkpoints
4. List and manage saved sessions
"""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

from genetic_mcp.models import GenerationRequest, EvolutionMode, FitnessWeights
from genetic_mcp.session_manager import SessionManager
from genetic_mcp.llm_client import MultiModelClient, OpenAIClient


async def persistence_example():
    """Demonstrate persistence system functionality."""
    print("🔄 Genetic MCP Persistence System Example")
    print("=" * 50)
    
    # Initialize LLM client (you'll need actual API keys for full functionality)
    llm_client = MultiModelClient(
        openai_client=OpenAIClient("fake-key-for-demo"),  # Replace with real key
        default_provider="openai"
    )
    
    # Initialize session manager with persistence enabled
    session_manager = SessionManager(
        llm_client=llm_client,
        persistence_db_path="demo_sessions.db",
        enable_auto_save=True  # Auto-save every 3 minutes
    )
    
    await session_manager.start()
    print("✅ Session manager started with persistence enabled")
    
    try:
        # Example 1: Create and save a session
        print("\n📝 Example 1: Creating and Saving a Session")
        print("-" * 40)
        
        request = GenerationRequest(
            prompt="Generate innovative features for a task management app",
            mode=EvolutionMode.ITERATIVE,
            population_size=8,
            generations=3,
            fitness_weights=FitnessWeights(
                relevance=0.5,
                novelty=0.3,
                feasibility=0.2
            ),
            client_generated=True,  # We'll inject ideas manually for demo
            use_memory_system=True
        )
        
        session = await session_manager.create_session("demo-client", request)
        print(f"✅ Created session: {session.id}")
        
        # Manually save the session
        success = await session_manager.save_session_to_db(session.id, "initial_creation")
        print(f"✅ Session saved to database: {success}")
        
        # Example 2: Create a checkpoint
        print("\n🛡️ Example 2: Creating a Checkpoint")
        print("-" * 40)
        
        # Simulate some progress
        session.current_generation = 1
        session.status = "active"
        
        success = await session_manager.save_checkpoint(
            session.id, 
            "generation_1_start",
            {"note": "Starting generation 1", "population_fitness": [0.7, 0.8, 0.6]}
        )
        print(f"✅ Checkpoint saved: {success}")
        
        # Example 3: List saved sessions
        print("\n📋 Example 3: Listing Saved Sessions")
        print("-" * 40)
        
        sessions = await session_manager.list_saved_sessions(limit=10)
        print(f"Found {len(sessions)} saved sessions:")
        for session_info in sessions:
            print(f"  - {session_info['id'][:12]}... | {session_info['prompt'][:50]}...")
            print(f"    Client: {session_info['client_id']} | Status: {session_info['status']} | Ideas: {session_info['idea_count']}")
        
        # Example 4: Load and resume a session
        print("\n🔄 Example 4: Loading and Resuming a Session")
        print("-" * 40)
        
        # First, let's "lose" the session by removing it from memory
        original_session_id = session.id
        await session_manager.delete_session(session.id)  # Remove from active sessions
        print("✅ Session removed from active memory")
        
        # Now load it back from database
        loaded_session = await session_manager.load_session_from_db(original_session_id)
        if loaded_session:
            print(f"✅ Session loaded from database: {loaded_session.id}")
            print(f"   Prompt: {loaded_session.prompt}")
            print(f"   Current generation: {loaded_session.current_generation}")
            print(f"   Status: {loaded_session.status}")
        
        # Resume the session (makes it active again)
        success = await session_manager.resume_session(original_session_id)
        print(f"✅ Session resumed: {success}")
        
        # Verify it's active again
        active_session = await session_manager.get_session(original_session_id)
        if active_session:
            print(f"✅ Session is now active with status: {active_session.status}")
        
        # Example 5: Working with multiple sessions
        print("\n📊 Example 5: Managing Multiple Sessions")
        print("-" * 40)
        
        # Create a few more sessions for demonstration
        session_ids = []
        prompts = [
            "Design features for a social media app",
            "Create innovative e-commerce solutions", 
            "Generate ideas for productivity tools"
        ]
        
        for i, prompt in enumerate(prompts):
            req = GenerationRequest(
                prompt=prompt,
                population_size=5,
                generations=2,
                client_generated=True
            )
            sess = await session_manager.create_session(f"client-{i}", req)
            session_ids.append(sess.id)
            
            # Save with different checkpoints
            await session_manager.save_session_to_db(sess.id, f"batch_created_{i}")
        
        print(f"✅ Created and saved {len(session_ids)} additional sessions")
        
        # List all sessions
        all_sessions = await session_manager.list_saved_sessions(limit=20)
        print(f"✅ Total sessions in database: {len(all_sessions)}")
        
        # Group by client
        by_client = {}
        for sess in all_sessions:
            client = sess["client_id"]
            if client not in by_client:
                by_client[client] = []
            by_client[client].append(sess)
        
        print("Sessions by client:")
        for client, sessions in by_client.items():
            print(f"  - {client}: {len(sessions)} sessions")
        
        # Example 6: Database statistics
        print("\n📈 Example 6: Database Statistics")
        print("-" * 40)
        
        stats = await session_manager.persistence_manager.get_database_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            if key.endswith("_count"):
                print(f"  - {key.replace('_', ' ').title()}: {value}")
            elif key == "database_size_bytes":
                size_kb = value / 1024
                print(f"  - Database Size: {size_kb:.1f} KB")
            else:
                print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 Persistence example completed successfully!")
        print("=" * 50)
        
    finally:
        # Cleanup
        await session_manager.stop()
        print("✅ Session manager stopped")
        
        # Note: The database file 'demo_sessions.db' will persist
        # You can examine it or delete it manually


def main():
    """Run the persistence example."""
    try:
        asyncio.run(persistence_example())
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()