"""
Architecture Diagram Generator for Genetic MCP Server

This script generates a visual representation of the system architecture
using matplotlib and networkx for component relationships.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Genetic Algorithm MCP Server Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Define component positions and styles
    components = {
        # Client Layer
        'client': {'pos': (2, 9), 'size': (3, 1.5), 'color': '#E8F4FD', 'label': 'MCP Client\n(Host Application)'},
        
        # Transport Layer
        'transport': {'pos': (7, 9), 'size': (2, 1.5), 'color': '#D4E6F1', 'label': 'Transport Layer\n(stdio/HTTP/SSE)'},
        
        # Core Server Components
        'mcp_server': {'pos': (11, 9), 'size': (3, 1.5), 'color': '#AED6F1', 'label': 'MCP Server Core\n(Protocol Handler)'},
        
        # Session Management
        'session_mgr': {'pos': (2, 6.5), 'size': (2.5, 1.2), 'color': '#85C1E2', 'label': 'Session\nManager'},
        'state_store': {'pos': (0.5, 4.5), 'size': (2, 1), 'color': '#F8D7DA', 'label': 'State Store\n(Redis)'},
        
        # Worker Management
        'worker_pool': {'pos': (5.5, 6.5), 'size': (2.5, 1.2), 'color': '#85C1E2', 'label': 'Worker Pool\nManager'},
        'llm_workers': {'pos': (5.5, 4.5), 'size': (2.5, 1), 'color': '#D1F2EB', 'label': 'LLM Workers\n(Parallel)'},
        
        # Genetic Algorithm
        'ga_engine': {'pos': (9, 6.5), 'size': (2.5, 1.2), 'color': '#85C1E2', 'label': 'Genetic\nEngine'},
        'population': {'pos': (9, 4.5), 'size': (2.5, 1), 'color': '#FCF3CF', 'label': 'Population\nStorage'},
        
        # Evaluation
        'fitness_eval': {'pos': (12.5, 6.5), 'size': (2.5, 1.2), 'color': '#85C1E2', 'label': 'Fitness\nEvaluator'},
        'metrics': {'pos': (12.5, 4.5), 'size': (2.5, 1), 'color': '#FADBD8', 'label': 'Metrics\n(R/N/F)'},
        
        # Supporting Components
        'idea_codec': {'pos': (3, 2.5), 'size': (2, 1), 'color': '#D5DBDB', 'label': 'Idea\nCodec'},
        'lineage': {'pos': (6, 2.5), 'size': (2, 1), 'color': '#D5DBDB', 'label': 'Lineage\nTracker'},
        'rate_limiter': {'pos': (9, 2.5), 'size': (2, 1), 'color': '#D5DBDB', 'label': 'Rate\nLimiter'},
        'monitor': {'pos': (12, 2.5), 'size': (2, 1), 'color': '#D5DBDB', 'label': 'Monitoring\n(Metrics)'},
    }
    
    # Draw components
    for comp_id, comp in components.items():
        box = FancyBboxPatch(
            comp['pos'], comp['size'][0], comp['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add label
        x, y = comp['pos']
        w, h = comp['size']
        ax.text(x + w/2, y + h/2, comp['label'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Define connections
    connections = [
        # Client flow
        ('client', 'transport', 'Requests'),
        ('transport', 'mcp_server', 'Messages'),
        
        # Server to components
        ('mcp_server', 'session_mgr', ''),
        ('mcp_server', 'worker_pool', ''),
        ('mcp_server', 'ga_engine', ''),
        ('mcp_server', 'fitness_eval', ''),
        
        # Session management
        ('session_mgr', 'state_store', 'Persist'),
        
        # Worker management
        ('worker_pool', 'llm_workers', 'Tasks'),
        ('worker_pool', 'rate_limiter', ''),
        
        # GA operations
        ('ga_engine', 'population', 'Store'),
        ('ga_engine', 'idea_codec', ''),
        ('ga_engine', 'lineage', 'Track'),
        
        # Fitness evaluation
        ('fitness_eval', 'metrics', 'Compute'),
        ('fitness_eval', 'llm_workers', 'Critic'),
        
        # Monitoring
        ('worker_pool', 'monitor', ''),
        ('ga_engine', 'monitor', ''),
        ('fitness_eval', 'monitor', ''),
    ]
    
    # Draw connections
    for start, end, label in connections:
        start_comp = components[start]
        end_comp = components[end]
        
        # Calculate connection points
        start_x = start_comp['pos'][0] + start_comp['size'][0]/2
        start_y = start_comp['pos'][1] + start_comp['size'][1]/2
        end_x = end_comp['pos'][0] + end_comp['size'][0]/2
        end_y = end_comp['pos'][1] + end_comp['size'][1]/2
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            connectionstyle="arc3,rad=0.1",
            arrowstyle="->",
            mutation_scale=20,
            linewidth=1,
            color='gray'
        )
        ax.add_patch(arrow)
        
        # Add label if provided
        if label:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add data flow indicators
    flow_boxes = [
        {'pos': (0.5, 0.5), 'text': 'Idea Generation Flow', 'color': '#E8F5E9'},
        {'pos': (4, 0.5), 'text': 'Evolution Cycle', 'color': '#FFF3E0'},
        {'pos': (7.5, 0.5), 'text': 'Fitness Evaluation', 'color': '#FCE4EC'},
        {'pos': (11.5, 0.5), 'text': 'Progress Streaming', 'color': '#F3E5F5'},
    ]
    
    for flow in flow_boxes:
        box = FancyBboxPatch(
            flow['pos'], 3, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=flow['color'],
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(flow['pos'][0] + 1.5, flow['pos'][1] + 0.4, flow['text'],
                ha='center', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('/home/andras/genetic_mcp/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/andras/genetic_mcp/architecture_diagram.pdf', bbox_inches='tight')
    print("Architecture diagram saved as architecture_diagram.png and architecture_diagram.pdf")

if __name__ == "__main__":
    create_architecture_diagram()