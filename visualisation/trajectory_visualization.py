"""
Trajectory Visualization Module for AI Agent Trajectory Anomaly Detection.

Handles:
- Trajectory graph/DAG plots
- Enhanced trajectory visualizations
- Node/edge type distributions
- Agent timelines
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from modules.utils import ensure_directory

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


class TrajectoryVisualizer:
    def __init__(self, config: Dict, output_dir: str = "charts"):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        ensure_directory(str(self.output_dir))
        
        # Setup matplotlib
        setup_matplotlib_for_plotting()
        
        # Color schemes
        self.color_schemes = self.viz_config.get('color_schemes', {})
        self.node_colors = {
            'user_enquiry': '#ff6b6b',  # Red for user interactions
            'user_interaction': '#ff8e8e',
            'llm_call': '#4ecdc4',  # Teal for LLM calls
            'tool_call': '#ff7f0e',
            'reasoning': '#2ca02c',
            'memory_access': '#d62728',
            'planning': '#9467bd',
            'validation': '#8c564b',
            'handoff': '#e377c2',
            'observation': '#7f7f7f',
            'action': '#bcbd22'
        }
        
        self.agent_colors = {
            'Planner': '#1f77b4',
            'Executor': '#ff7f0e',
            'Validator': '#2ca02c',
            'Coordinator': '#d62728'
        }
        
        self.severity_colors = {
            'low': '#ffffcc',
            'medium': '#feb24c',
            'high': '#fd8d3c',
            'critical': '#e31a1c'
        }
        
        # Figure settings
        self.figure_settings = self.viz_config.get('figure_settings', {})
        self.dpi = self.figure_settings.get('dpi', 300)
        self.format = self.figure_settings.get('format', 'png')
        self.bbox_inches = self.figure_settings.get('bbox_inches', 'tight')
        
        logger.info("TrajectoryVisualizer initialized, output dir: %s", self.output_dir)

    def plot_trajectory_examples(self, graphs: List[nx.DiGraph], 
                                normal_indices: List[int], anomalous_indices: List[int],
                                max_examples: int = 6) -> str:
        try:
            logger.info("Plotting trajectory examples")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            normal_examples = normal_indices[:max_examples//2]
            anomalous_examples = anomalous_indices[:max_examples//2]
            if not normal_examples and not anomalous_examples:
                logger.warning("No examples to plot")
                return ""
            total_examples = len(normal_examples) + len(anomalous_examples)
            cols = min(3, total_examples)
            rows = (total_examples + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if total_examples == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            for i, idx in enumerate(normal_examples):
                if idx < len(graphs):
                    self._plot_single_trajectory(graphs[idx], axes[i], f"Normal {idx}")
            for i, idx in enumerate(anomalous_examples):
                plot_idx = len(normal_examples) + i
                if idx < len(graphs):
                    self._plot_single_trajectory(graphs[idx], axes[plot_idx], f"Anomalous {idx}")
            for i in range(total_examples, len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            filepath = self.output_dir / f"trajectory_examples.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            logger.info("Saved trajectory examples to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.error(f"Error in plot_trajectory_examples: {e}")
            return ""

    def _plot_single_trajectory(self, graph: nx.DiGraph, ax, title: str) -> None:
        """Plot a single trajectory graph with improved readability, floating node detection, and node numbering."""
        if len(graph.nodes) == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Choose layout based on graph structure with better spacing
        if len(graph.nodes) < 20:
            try:
                # Use spring layout with increased spacing for better readability
                pos = nx.spring_layout(graph, k=4, iterations=200, scale=2.0)  # Increased k and scale
            except:
                try:
                    pos = nx.kamada_kawai_layout(graph, scale=2.0)  # Increased scale
                except:
                    pos = nx.random_layout(graph)
        else:
            try:
                pos = nx.spring_layout(graph, k=5, iterations=200, scale=3.0)  # Even more spacing for large graphs
            except:
                pos = nx.kamada_kawai_layout(graph, scale=3.0)
        
        # Check for floating nodes (disconnected)
        components = list(nx.weakly_connected_components(graph))
        if len(components) > 1:
            logger.warning(f"Floating nodes detected in {title}: {components}")
            # Highlight floating nodes
            floating_nodes = [n for comp in components[1:] for n in comp]
            for node in floating_nodes:
                if node in pos:
                    # Draw floating nodes in black with yellow border
                    nx.draw_networkx_nodes(
                        graph, pos, nodelist=[node],
                        node_color='black', node_size=400, alpha=0.8, ax=ax,
                        edgecolors='yellow', linewidths=3
                    )
        
        # Identify start and end nodes
        start_nodes = [node for node, data in graph.nodes(data=True) 
                      if data.get('node_type') == 'user_enquiry']
        end_nodes = [node for node, data in graph.nodes(data=True) 
                    if data.get('node_type') == 'action']
        
        # Generate node numbers based on execution flow
        node_numbers = self._generate_node_numbers(graph)
        
        # Draw nodes colored by type with increased size for better visibility
        for node_type in self.node_colors:
            node_list = [node for node, data in graph.nodes(data=True) 
                        if data.get('node_type') == node_type]
            if node_list:
                # Special styling for start and end nodes
                if node_type == 'user_enquiry':
                    # Start nodes: larger, with border
                    nx.draw_networkx_nodes(
                        graph, pos, nodelist=node_list,
                        node_color=self.node_colors[node_type],
                        node_size=600, alpha=0.9, ax=ax,  # Increased size
                        edgecolors='green', linewidths=3
                    )
                elif node_type == 'action':
                    # End nodes: larger, with border
                    nx.draw_networkx_nodes(
                        graph, pos, nodelist=node_list,
                        node_color=self.node_colors[node_type],
                        node_size=600, alpha=0.9, ax=ax,  # Increased size
                        edgecolors='red', linewidths=3
                    )
                else:
                    nx.draw_networkx_nodes(
                        graph, pos, nodelist=node_list,
                        node_color=self.node_colors[node_type],
                        node_size=400, alpha=0.8, ax=ax  # Increased size
                    )
        
        # Draw other nodes in default color
        other_nodes = [node for node, data in graph.nodes(data=True)
                      if data.get('node_type') not in self.node_colors]
        if other_nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=other_nodes,
                node_color='lightgray', node_size=400, alpha=0.8, ax=ax  # Increased size
            )
        
        # Draw edges with curved style for better readability and no double arrows
        normal_edges = [(u, v) for u, v, data in graph.edges(data=True)
                       if not data.get('is_anomalous', False)]
        anomalous_edges = [(u, v) for u, v, data in graph.edges(data=True)
                          if data.get('is_anomalous', False)]
        
        if normal_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=normal_edges,
                edge_color='gray', alpha=0.6, arrows=True, ax=ax,
                arrowsize=20, arrowstyle='->', width=2,  # Increased arrow size and width
                connectionstyle='arc3,rad=0.15'  # More curved edges
            )
        
        if anomalous_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=anomalous_edges,
                edge_color='red', alpha=0.8, arrows=True, width=4, ax=ax,  # Increased width
                arrowsize=25, arrowstyle='->',  # Increased arrow size
                connectionstyle='arc3,rad=0.25'  # More curved for anomalous edges
            )
        
        # Create meaningful node labels with numbers
        labels = {}
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('node_type', '')
            agent_type = data.get('agent_type', '')
            description = data.get('description', '')
            
            # Get node number
            node_num = node_numbers.get(node_id, '?')
            
            # Create meaningful labels based on node type and content
            if node_type == 'user_enquiry':
                if description:
                    # Truncate description for readability
                    short_desc = description[:25] + "..." if len(description) > 25 else description
                    labels[node_id] = f"{node_num}. START\n{short_desc}"
                else:
                    labels[node_id] = f"{node_num}. START\nUser Enquiry"
            elif node_type == 'action':
                if description:
                    labels[node_id] = f"{node_num}. END\n{description[:25]}..."
                else:
                    labels[node_id] = f"{node_num}. END\nAction"
            elif node_type == 'llm_call':
                llm_purpose = data.get('llm_purpose', '')
                if llm_purpose:
                    labels[node_id] = f"{node_num}. LLM\n{llm_purpose[:20]}..."
                elif description:
                    labels[node_id] = f"{node_num}. LLM\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_num}. LLM\nCall"
            elif node_type == 'handoff':
                if description:
                    labels[node_id] = f"{node_num}. Handoff\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_num}. Handoff\n{agent_type}"
            elif node_type == 'tool_call':
                tool_type = data.get('tool_type', '')
                if tool_type:
                    labels[node_id] = f"{node_num}. Tool\n{tool_type[:15]}..."
                elif description:
                    labels[node_id] = f"{node_num}. Tool\n{description[:15]}..."
                else:
                    labels[node_id] = f"{node_num}. Tool\nCall"
            elif node_type == 'planning':
                if description:
                    labels[node_id] = f"{node_num}. Plan\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_num}. Planning\n{agent_type}"
            elif node_type == 'reasoning':
                if description:
                    labels[node_id] = f"{node_num}. Reason\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_num}. Reasoning\n{agent_type}"
            elif node_type == 'validation':
                if description:
                    labels[node_id] = f"{node_num}. Validate\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_num}. Validation\n{agent_type}"
            else:
                # For other node types, use description or type
                if description:
                    labels[node_id] = f"{node_num}. {node_type[:8]}...\n{description[:15]}..."
                else:
                    labels[node_id] = f"{node_num}. {node_type[:8]}...\n{agent_type}"
        
        # Draw labels with better formatting and backgrounds
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax, 
                               font_weight='bold', font_family='sans-serif',
                               bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray'))  # Increased padding
        
        ax.set_title(title, fontsize=12, fontweight='bold')  # Increased title size
        ax.axis('off')

    def _generate_node_numbers(self, graph: nx.DiGraph) -> Dict[str, str]:
        """Generate clear step numbers based on execution flow, ensuring sequential progression."""
        if len(graph.nodes) == 0:
            return {}
        
        # Find start nodes (nodes with no incoming edges or user_enquiry type)
        start_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        if not start_nodes:
            # If no clear start nodes, use nodes with user_enquiry type
            start_nodes = [node for node, data in graph.nodes(data=True) 
                          if data.get('node_type') == 'user_enquiry']
        
        if not start_nodes:
            # If still no start nodes, use any node
            start_nodes = list(graph.nodes())[:1]
        
        # Use topological sort to get execution order
        try:
            # Get topological sort
            topo_order = list(nx.topological_sort(graph))
            node_numbers = {}
            
            # Assign numbers based on topological order
            for i, node in enumerate(topo_order, 1):
                node_numbers[node] = str(i)
            
            return node_numbers
            
        except nx.NetworkXError:
            # If topological sort fails (e.g., cycles), use BFS from start nodes
            logger.warning("Topological sort failed, using BFS for node numbering")
            node_numbers = {}
            visited = set()
            queue = [(node, 1) for node in start_nodes]
            
            while queue:
                node, number = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    node_numbers[node] = str(number)
                    
                    # Add neighbors to queue
                    for neighbor in graph.successors(node):
                        if neighbor not in visited:
                            queue.append((neighbor, number + 1))
            
            # Handle any remaining nodes
            for node in graph.nodes():
                if node not in node_numbers:
                    node_numbers[node] = str(len(node_numbers) + 1)
            
            return node_numbers

    def plot_enhanced_trajectory(self, trajectory, output_path: str = None) -> str:
        """
        Create an enhanced trajectory visualization with detailed node information and error/success outcomes.
        
        Args:
            trajectory: AgentTrajectory object
            output_path: Optional output path for the figure (filename only)
        
        Returns:
            Path to saved figure
        """
        logger.info("Creating enhanced trajectory visualization")
        
        try:
            # Debug logging to identify the issue
            logger.debug(f"Trajectory ID: {getattr(trajectory, 'trajectory_id', 'unknown')}")
            logger.debug(f"Trajectory is_anomalous: {getattr(trajectory, 'is_anomalous', 'unknown')}")
            logger.debug(f"Trajectory has anomaly_severity: {hasattr(trajectory, 'anomaly_severity')}")
            
            # Create NetworkX graph from trajectory
            G = nx.DiGraph()
            
            # Add nodes with enhanced attributes
            for node in trajectory.nodes:
                G.add_node(node.node_id,
                          node_type=node.node_type.value,
                          agent_type=node.agent_type.value if node.agent_type else None,
                          description=node.description,
                          timestamp=getattr(node, 'timestamp', None),  # Handle missing timestamp
                          duration=node.duration,
                          llm_purpose=node.llm_purpose,
                          tool_type=node.tool_type.value if node.tool_type else None,
                          # Add error/success information
                          is_failed=node.is_failed(),
                          status=node.status,
                          error_code=node.error_code,
                          error_message=node.error_message,
                          tool_success=node.tool_success,
                          memory_success=node.memory_success,
                          handoff_success=node.handoff_success,
                          validation_result=node.validation_result)
            
            # Add edges
            for edge in trajectory.edges:
                G.add_edge(edge.source_node_id, edge.target_node_id,
                          edge_type=edge.edge_type,
                          relationship_type=edge.relationship_type,
                          # Add edge error/success information
                          is_failed=edge.is_failed(),
                          success_rate=edge.success_rate,
                          error_count=edge.error_count)
            
            # Create outcome nodes for each action node
            outcome_nodes = {}
            # Collect nodes first to avoid iteration error
            nodes_to_process = list(G.nodes(data=True))
            for node_id, data in nodes_to_process:
                if data.get('node_type') in ['tool_call', 'llm_call', 'handoff', 'validation', 'memory_access']:
                    # Create outcome node ID
                    outcome_id = f"{node_id}_outcome"
                    outcome_nodes[node_id] = outcome_id
                    
                    # Determine outcome status
                    is_failed = data.get('is_failed', False)
                    status = data.get('status', 'unknown')
                    
                    # Add outcome node
                    G.add_node(outcome_id,
                              node_type='outcome',
                              description=f"Status: {status}",
                              is_failed=is_failed,
                              outcome_type='success' if not is_failed else 'failure')
                    
                    # Add edge to outcome
                    G.add_edge(node_id, outcome_id, edge_type='outcome')
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1], width_ratios=[2, 1, 1])
            
            # Main trajectory graph
            ax_main = fig.add_subplot(gs[0, :])
            self._plot_enhanced_trajectory_graph(G, ax_main, trajectory)
            
            # Statistics panel
            ax_stats = fig.add_subplot(gs[1, 0])
            self._plot_trajectory_statistics(trajectory, ax_stats)
            
            # Agent timeline
            ax_timeline = fig.add_subplot(gs[1, 1])
            self._plot_agent_timeline(trajectory, ax_timeline)
            
            # Node type distribution
            ax_dist = fig.add_subplot(gs[1, 2])
            self._plot_node_type_distribution(trajectory, ax_dist)
            
            # Error analysis
            ax_errors = fig.add_subplot(gs[2, :])
            self._plot_error_analysis(trajectory, ax_errors)
            
            # Performance metrics
            ax_perf = fig.add_subplot(gs[3, :])
            self._plot_performance_metrics(trajectory, ax_perf)
            
            plt.tight_layout()
            
            # Always save in a 'trajectories' subdirectory
            traj_dir = self.output_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            if output_path:
                filepath = traj_dir / f"{output_path}.{self.format}"
            else:
                filepath = traj_dir / f"enhanced_trajectory_{getattr(trajectory, 'trajectory_id', 'unknown')}.{self.format}"
            
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved enhanced trajectory visualization to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_enhanced_trajectory: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def _plot_enhanced_trajectory_graph(self, G: nx.DiGraph, ax, trajectory) -> None:
        """Plot the main trajectory graph with enhanced styling."""
        try:
            if len(G.nodes) == 0:
                ax.text(0.5, 0.5, "Empty Trajectory", ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Enhanced Trajectory Visualization")
                return
            
            # Use hierarchical layout for better flow visualization
            try:
                pos = nx.spring_layout(G, k=3, iterations=100, scale=2.0)
            except:
                pos = nx.kamada_kawai_layout(G, scale=2.0)
            
            # Draw nodes by type with enhanced styling
            for node_type in self.node_colors:
                node_list = [node for node, data in G.nodes(data=True) 
                            if data.get('node_type') == node_type]
                if node_list:
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=node_list,
                        node_color=self.node_colors[node_type],
                        node_size=500, alpha=0.8, ax=ax
                    )
            
            # Draw outcome nodes
            outcome_nodes = [node for node, data in G.nodes(data=True) 
                           if data.get('node_type') == 'outcome']
            if outcome_nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=outcome_nodes,
                    node_color='lightgreen', node_size=300, alpha=0.7, ax=ax,
                    node_shape='s'  # Square shape for outcomes
                )
            
            # Draw edges
            normal_edges = [(u, v) for u, v, data in G.edges(data=True)
                           if data.get('edge_type') != 'outcome']
            outcome_edges = [(u, v) for u, v, data in G.edges(data=True)
                            if data.get('edge_type') == 'outcome']
            
            if normal_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=normal_edges,
                    edge_color='gray', alpha=0.6, arrows=True, ax=ax,
                    arrowsize=15, arrowstyle='->', width=1.5
                )
            
            if outcome_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=outcome_edges,
                    edge_color='green', alpha=0.8, arrows=True, ax=ax,
                    arrowsize=10, arrowstyle='->', width=2,
                    connectionstyle='arc3,rad=0.3'
                )
            
            # Create labels
            labels = {}
            for node_id, data in G.nodes(data=True):
                node_type = data.get('node_type', '')
                description = data.get('description', '')
                
                if node_type == 'outcome':
                    outcome_type = data.get('outcome_type', 'unknown')
                    labels[node_id] = f"✓" if outcome_type == 'success' else f"✗"
                elif description:
                    labels[node_id] = f"{node_type[:8]}...\n{description[:20]}..."
                else:
                    labels[node_id] = f"{node_type[:8]}..."
            
            nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_title(f"Enhanced Trajectory Visualization - {'Anomalous' if getattr(trajectory, 'is_anomalous', False) else 'Normal'}", 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            logger.error(f"Error in _plot_enhanced_trajectory_graph: {e}")

    def _plot_trajectory_statistics(self, trajectory, ax):
        """Plot trajectory statistics."""
        try:
            stats = {
                'Total Nodes': len(trajectory.nodes),
                'Total Edges': len(trajectory.edges),
                'Duration (s)': getattr(trajectory, 'total_duration', 0),
                'Completion Rate': getattr(trajectory, 'completion_rate', 0),
                'Success': getattr(trajectory, 'success', False),
                'Anomalous': getattr(trajectory, 'is_anomalous', False)
            }
            
            # Create bar chart
            keys = list(stats.keys())
            values = list(stats.values())
            colors = ['blue' if v != True else 'green' if v else 'red' for v in values]
            
            bars = ax.bar(keys, values, color=colors, alpha=0.7)
            ax.set_title('Trajectory Statistics')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if isinstance(value, bool):
                    label = 'Yes' if value else 'No'
                elif isinstance(value, float):
                    label = f'{value:.2f}'
                else:
                    label = str(value)
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       label, ha='center', va='bottom')
            
        except Exception as e:
            logger.error(f"Error in _plot_trajectory_statistics: {e}")

    def _plot_agent_timeline(self, trajectory, ax):
        """Plot agent activity timeline."""
        try:
            agent_times = {}
            colors = list(self.agent_colors.values())
            for node in trajectory.nodes:
                agent = node.agent_type.value if node.agent_type else 'Unknown'
                if agent not in agent_times:
                    agent_times[agent] = []
                # Only add timestamp if it exists and is not None
                if hasattr(node, 'timestamp') and node.timestamp is not None:
                    agent_times[agent].append(node.timestamp)
            for i, (agent, times) in enumerate(agent_times.items()):
                if times:
                    ax.scatter(times, [agent] * len(times), 
                              c=colors[i % len(colors)], s=100, alpha=0.7, label=agent)
            ax.set_title('Agent Activity Timeline')
            ax.set_xlabel('Time')
            ax.set_ylabel('Agent Type')
            ax.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.error(f"Error in _plot_agent_timeline: {e}")

    def _plot_node_type_distribution(self, trajectory, ax):
        """Plot distribution of node types."""
        try:
            node_types = {}
            for node in trajectory.nodes:
                node_type = node.node_type.value
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_types:
                colors = [self.node_colors.get(nt, '#cccccc') for nt in node_types.keys()]
                wedges, texts, autotexts = ax.pie(node_types.values(), labels=node_types.keys(), 
                                                 autopct='%1.1f%%', colors=colors)
                ax.set_title('Node Type Distribution')
        except Exception as e:
            logger.error(f"Error in _plot_node_type_distribution: {e}")

    def _plot_error_analysis(self, trajectory, ax):
        """Plot error analysis."""
        try:
            error_counts = {
                'Failed Nodes': sum(1 for node in trajectory.nodes if node.is_failed()),
                'Failed Edges': sum(1 for edge in trajectory.edges if edge.is_failed()),
                'Tool Failures': sum(1 for node in trajectory.nodes if not getattr(node, 'tool_success', True)),
                'Memory Failures': sum(1 for node in trajectory.nodes if not getattr(node, 'memory_success', True)),
                'Handoff Failures': sum(1 for node in trajectory.nodes if not getattr(node, 'handoff_success', True))
            }
            
            if any(error_counts.values()):
                keys = list(error_counts.keys())
                values = list(error_counts.values())
                bars = ax.bar(keys, values, color='red', alpha=0.7)
                ax.set_title('Error Analysis')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           str(value), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Errors Detected', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Error Analysis')
            
        except Exception as e:
            logger.error(f"Error in _plot_error_analysis: {e}")

    def _plot_performance_metrics(self, trajectory, ax):
        """Plot performance metrics."""
        try:
            metrics = {
                'Completion Rate': getattr(trajectory, 'completion_rate', 0),
                'Success Rate': 1.0 if getattr(trajectory, 'success', False) else 0.0,
                'Efficiency': getattr(trajectory, 'efficiency', 0),
                'Quality Score': getattr(trajectory, 'quality_score', 0)
            }
            
            keys = list(metrics.keys())
            values = list(metrics.values())
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
            
            bars = ax.bar(keys, values, color=colors, alpha=0.7)
            ax.set_title('Performance Metrics')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
            
        except Exception as e:
            logger.error(f"Error in _plot_performance_metrics: {e}") 