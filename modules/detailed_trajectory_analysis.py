#!/usr/bin/env python3
"""
Detailed trajectory analysis visualization module.
Creates comprehensive analysis of individual trajectory graphs.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

class DetailedTrajectoryAnalysis:
    """
    Creates detailed analysis visualizations for individual trajectory graphs.
    """
    
    def __init__(self, output_dir: Path, format: str = 'png', dpi: int = 300, 
                 bbox_inches: str = 'tight'):
        """
        Initialize the detailed trajectory analysis visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            format: Image format (png, pdf, etc.)
            dpi: Image resolution
            bbox_inches: Bounding box setting
        """
        self.output_dir = output_dir
        self.format = format
        self.dpi = dpi
        self.bbox_inches = bbox_inches
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_detailed_trajectory_analysis(self, 
                                          graphs: List[nx.DiGraph],
                                          features_df: pd.DataFrame,
                                          model_results: Dict[str, Any],
                                          test_df: pd.DataFrame) -> str:
        """
        Create detailed analysis for selected normal and anomalous trajectories.
        
        Args:
            graphs: List of NetworkX graphs representing trajectories
            features_df: DataFrame with trajectory features
            model_results: Dictionary containing model results and predictions
            test_df: Test DataFrame with labels
            
        Returns:
            Path to the detailed analysis folder
        """
        try:
            logger.info("Creating detailed trajectory analysis")
            
            # Convert test_df to a simple list to avoid pandas Series issues
            test_labels = []
            for i in range(len(test_df)):
                try:
                    label = test_df.iloc[i].get('is_anomalous', 0)
                    if pd.isna(label):
                        label = 0
                    test_labels.append(int(label))
                except:
                    test_labels.append(0)
            
            # Simple approach: select first 3 normal and 3 anomalous trajectories
            normal_count = 0
            anomalous_count = 0
            normal_graphs = []
            anomalous_graphs = []
            
            # Iterate through graphs and separate them
            for i, graph in enumerate(graphs):
                if i < len(test_labels):
                    is_anomalous = test_labels[i]
                    
                    if is_anomalous == 0 and normal_count < 3:
                        normal_graphs.append((i, graph))
                        normal_count += 1
                    elif is_anomalous == 1 and anomalous_count < 3:
                        anomalous_graphs.append((i, graph))
                        anomalous_count += 1
                    
                    if normal_count >= 3 and anomalous_count >= 3:
                        break
            
            logger.info(f"Selected {len(normal_graphs)} normal and {len(anomalous_graphs)} anomalous trajectories")
            
            # Create analysis for normal trajectories
            for i, (idx, graph) in enumerate(normal_graphs):
                if idx < len(features_df) and idx < len(test_df):
                    self._create_simple_trajectory_analysis(
                        graph, features_df.iloc[idx], test_df.iloc[idx], 
                        f"normal_trajectory_{i+1}", "Normal"
                    )
            
            # Create analysis for anomalous trajectories
            for i, (idx, graph) in enumerate(anomalous_graphs):
                if idx < len(features_df) and idx < len(test_df):
                    self._create_simple_trajectory_analysis(
                        graph, features_df.iloc[idx], test_df.iloc[idx], 
                        f"anomalous_trajectory_{i+1}", "Anomalous"
                    )
            
            logger.info(f"Created detailed analysis for {len(normal_graphs) + len(anomalous_graphs)} trajectories")
            return str(self.output_dir)
            
        except Exception as e:
            logger.error(f"Error creating detailed trajectory analysis: {e}")
            return ""
    
    def _create_simple_trajectory_analysis(self, 
                                         graph: nx.DiGraph,
                                         features: pd.Series,
                                         test_row: pd.Series,
                                         filename: str,
                                         trajectory_type: str) -> None:
        """
        Create simple analysis for a single trajectory with graph and steps table.
        
        Args:
            graph: NetworkX graph representing the trajectory
            features: Feature series for this trajectory
            test_row: Test data row for this trajectory
            filename: Output filename
            trajectory_type: Type of trajectory (Normal/Anomalous)
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 8))
            
            # Create grid layout - 1 row, 2 columns
            gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
            
            # 1. Trajectory Graph Visualization
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_trajectory_graph(graph, ax1, trajectory_type)
            
            # 2. Trajectory Steps Table
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_trajectory_steps_table(graph, ax2)
            
            # Add title
            fig.suptitle(f'{trajectory_type} Trajectory Analysis - {filename}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save the detailed analysis
            filepath = self.output_dir / f"{filename}_detailed_analysis.{self.format}"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved detailed analysis for {filename}")
            
        except Exception as e:
            logger.error(f"Error creating detailed analysis for {filename}: {e}")
    
    def _create_single_trajectory_analysis(self, 
                                         trajectory: Any,
                                         features: pd.Series,
                                         model_results: Dict[str, Any],
                                         test_row: pd.Series,
                                         filename: str,
                                         trajectory_type: str) -> None:
        """
        Create detailed analysis for a single trajectory.
        
        Args:
            trajectory: Trajectory object
            features: Feature series for this trajectory
            model_results: Model results dictionary
            test_row: Test data row for this trajectory
            filename: Output filename
            trajectory_type: Type of trajectory (Normal/Anomalous)
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            
            # Create grid layout
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Trajectory Graph Visualization
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_trajectory_graph(trajectory, ax1, trajectory_type)
            
            # 2. Trajectory Steps Table
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_trajectory_steps_table(trajectory, ax2)
            
            # 3. Model Predictions
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_model_predictions(features, model_results, ax3)
            
            # 4. Key Statistics
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_key_statistics(features, test_row, ax4)
            
            # Add title
            fig.suptitle(f'{trajectory_type} Trajectory Analysis', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Save the detailed analysis
            filepath = self.output_dir / f"{filename}_detailed_analysis.{self.format}"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved detailed analysis for {filename}")
            
        except Exception as e:
            logger.error(f"Error creating detailed analysis for {filename}: {e}")
    
    def _plot_trajectory_graph(self, graph: nx.DiGraph, ax: plt.Axes, trajectory_type: str) -> None:
        """Plot the trajectory graph with node and edge information."""
        try:
            if graph is not None and len(graph.nodes()) > 0:
                # Create layout
                pos = nx.spring_layout(graph, k=2, iterations=100, scale=2.0)
                
                # Draw nodes with colors based on node type
                node_colors = []
                for node in graph.nodes():
                    node_data = graph.nodes[node]
                    node_type = node_data.get('node_type', 'unknown')
                    if node_type == 'user_enquiry':
                        node_colors.append('green')
                    elif node_type == 'action':
                        node_colors.append('red')
                    elif node_type == 'llm_call':
                        node_colors.append('blue')
                    elif node_type == 'tool_call':
                        node_colors.append('orange')
                    else:
                        node_colors.append('lightblue')
                
                # Draw nodes
                nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                                     node_size=400, ax=ax, alpha=0.8)
                
                # Draw edges
                nx.draw_networkx_edges(graph, pos, width=1.5, 
                                     edge_color='gray', alpha=0.7, ax=ax, arrows=True,
                                     arrowsize=15, arrowstyle='->')
                
                # Create node labels
                labels = {}
                for i, node in enumerate(graph.nodes()):
                    node_data = graph.nodes[node]
                    node_type = node_data.get('node_type', 'unknown')
                    
                    # Get meaningful description based on node type
                    description = ''
                    if node_type == 'user_enquiry':
                        description = node_data.get('description', 'User query')
                    elif node_type == 'llm_call':
                        llm_purpose = node_data.get('llm_purpose', '')
                        if llm_purpose:
                            description = f"LLM: {llm_purpose}"
                        else:
                            description = node_data.get('description', 'LLM call')
                    elif node_type == 'tool_call':
                        tool_type = node_data.get('tool_type', '')
                        if tool_type:
                            description = f"Tool: {tool_type}"
                        else:
                            description = node_data.get('description', 'Tool call')
                    elif node_type == 'action':
                        description = node_data.get('description', 'Action taken')
                    elif node_type == 'handoff':
                        description = node_data.get('description', 'Agent handoff')
                    elif node_type == 'planning':
                        description = node_data.get('description', 'Planning step')
                    elif node_type == 'reasoning':
                        description = node_data.get('description', 'Reasoning step')
                    elif node_type == 'validation':
                        description = node_data.get('description', 'Validation step')
                    elif node_type == 'memory_access':
                        description = node_data.get('description', 'Memory access')
                    elif node_type == 'outcome':
                        outcome_type = node_data.get('outcome_type', 'unknown')
                        status = node_data.get('description', '')
                        description = f"{outcome_type.upper()}: {status}"
                    else:
                        description = node_data.get('description', 'Unknown step')
                    
                    if description:
                        # Truncate description for readability
                        short_desc = description[:25] + "..." if len(description) > 25 else description
                        labels[node] = f"{i+1}. {node_type[:8]}\n{short_desc}"
                    else:
                        labels[node] = f"{i+1}. {node_type[:8]}"
                
                # Draw labels
                nx.draw_networkx_labels(graph, pos, labels, font_size=7, ax=ax,
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                ax.set_title(f'{trajectory_type} Trajectory Graph\n'
                           f'Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}')
                ax.axis('off')
                
            else:
                ax.text(0.5, 0.5, 'Empty graph', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trajectory Graph')
                ax.axis('off')
                
        except Exception as e:
            logger.error(f"Error plotting trajectory graph: {e}")
            ax.text(0.5, 0.5, 'Error plotting graph', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_trajectory_steps_table(self, graph: nx.DiGraph, ax: plt.Axes) -> None:
        """Plot trajectory steps in a table format."""
        try:
            if graph is not None and len(list(graph.nodes())) > 0:
                # Create table data from graph nodes
                table_data = []
                headers = ['Step', 'Node Type', 'Agent Type', 'Description']
                
                # Get nodes in order
                nodes_list = list(graph.nodes())
                for i, node in enumerate(nodes_list[:10]):  # Show first 10 steps
                    node_data = graph.nodes[node]
                    node_type = node_data.get('node_type', 'unknown')
                    agent_type = node_data.get('agent_type', 'N/A')
                    
                    # Get meaningful description based on node type
                    description = ''
                    if node_type == 'user_enquiry':
                        description = node_data.get('description', 'User query')
                    elif node_type == 'llm_call':
                        llm_purpose = node_data.get('llm_purpose', '')
                        if llm_purpose:
                            description = f"LLM: {llm_purpose}"
                        else:
                            description = node_data.get('description', 'LLM call')
                    elif node_type == 'tool_call':
                        tool_type = node_data.get('tool_type', '')
                        if tool_type:
                            description = f"Tool: {tool_type}"
                        else:
                            description = node_data.get('description', 'Tool call')
                    elif node_type == 'action':
                        description = node_data.get('description', 'Action taken')
                    elif node_type == 'handoff':
                        description = node_data.get('description', 'Agent handoff')
                    elif node_type == 'planning':
                        description = node_data.get('description', 'Planning step')
                    elif node_type == 'reasoning':
                        description = node_data.get('description', 'Reasoning step')
                    elif node_type == 'validation':
                        description = node_data.get('description', 'Validation step')
                    elif node_type == 'memory_access':
                        description = node_data.get('description', 'Memory access')
                    elif node_type == 'outcome':
                        outcome_type = node_data.get('outcome_type', 'unknown')
                        status = node_data.get('description', '')
                        description = f"{outcome_type.upper()}: {status}"
                    else:
                        description = node_data.get('description', 'Unknown step')
                    
                    # Truncate description for table
                    if description:
                        short_desc = description[:40] + "..." if len(description) > 40 else description
                    else:
                        short_desc = 'N/A'
                    
                    table_data.append([i+1, node_type, agent_type, short_desc])
                
                # Create table
                table = ax.table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 2)
                
                # Style the table
                for i in range(len(table_data) + 1):
                    for j in range(len(headers)):
                        cell = table[(i, j)]
                        if i == 0:  # Header
                            cell.set_facecolor('lightgray')
                            cell.set_text_props(weight='bold')
                        else:
                            cell.set_facecolor('white')
                
                ax.set_title('Trajectory Steps (First 10)')
                ax.axis('off')
                
            else:
                ax.text(0.5, 0.5, 'No graph data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trajectory Steps')
                
        except Exception as e:
            logger.error(f"Error plotting trajectory steps table: {e}")
            ax.text(0.5, 0.5, 'Error plotting steps', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_key_statistics(self, features: pd.Series, test_row: pd.Series, ax: plt.Axes) -> None:
        """Plot key statistics for the trajectory."""
        try:
            # Extract key statistics
            stats_data = {
                'Length': len(features),
                'Avg Speed': features.get('avg_speed', 0),
                'Max Speed': features.get('max_speed', 0),
                'Total Distance': features.get('total_distance', 0),
                'Duration': features.get('duration', 0),
                'Complexity': features.get('complexity_score', 0),
                'True Label': test_row.get('is_anomalous', 0),
                'Anomaly Severity': test_row.get('anomaly_severity', 0)
            }
            
            # Create table
            table_data = [[key, f'{value:.3f}' if isinstance(value, float) else str(value)] 
                         for key, value in stats_data.items()]
            
            table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                           cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('lightgray')
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('white')
            
            ax.set_title('Key Statistics')
            ax.axis('off')
            
        except Exception as e:
            logger.error(f"Error plotting key statistics: {e}")
            ax.text(0.5, 0.5, 'Error plotting statistics', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_model_predictions(self, features: pd.Series, model_results: Dict[str, Any], ax: plt.Axes) -> None:
        """Plot model predictions for this trajectory."""
        try:
            # Get predictions from different models
            predictions = {}
            
            for model_name, model_data in model_results.items():
                if model_name == 'ensemble_model':
                    continue
                    
                if 'model' in model_data and model_data['model'] is not None:
                    try:
                        # Get prediction for this trajectory
                        if hasattr(model_data['model'], 'decision_function'):
                            score = model_data['model'].decision_function([features.values])[0]
                        elif hasattr(model_data['model'], 'predict'):
                            score = model_data['model'].predict([features.values])[0]
                        else:
                            score = 0.0
                        
                        predictions[model_name] = score
                    except:
                        predictions[model_name] = 0.0
            
            if predictions:
                models = list(predictions.keys())
                scores = list(predictions.values())
                
                colors = ['red' if score < 0 else 'green' for score in scores]
                bars = ax.bar(models, scores, color=colors, alpha=0.7)
                ax.set_title('Model Predictions')
                ax.set_ylabel('Anomaly Score')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top')
            else:
                ax.text(0.5, 0.5, 'No model predictions available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Predictions')
                
        except Exception as e:
            logger.error(f"Error plotting model predictions: {e}")
            ax.text(0.5, 0.5, 'Error plotting predictions', 
                   ha='center', va='center', transform=ax.transAxes) 