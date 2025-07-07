"""
Anomaly & Data Analysis Visualization Module for AI Agent Trajectory Anomaly Detection.

Handles:
- Anomaly distribution
- Anomaly pattern analysis
- Synthetic data quality
- Feature importance
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


class AnomalyDataVisualizer:
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
        
        logger.info("AnomalyDataVisualizer initialized, output dir: %s", self.output_dir)

    def plot_anomaly_distribution(self, test_df: pd.DataFrame) -> str:
        try:
            logger.info("Plotting anomaly distribution")
            
            # Handle duplicate columns
            if test_df.columns.duplicated().any():
                logger.warning("Duplicate columns found in test_df, removing duplicates")
                test_df = test_df.loc[:, ~test_df.columns.duplicated()]
            
            if 'is_anomalous' not in test_df.columns:
                logger.warning("Column 'is_anomalous' not found in test_df")
                return ""
            
            anomalous_df = test_df[test_df['is_anomalous'] == True]
            if len(anomalous_df) == 0:
                logger.warning("No anomalous samples found")
                return ""
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            ax1 = axes[0]
            
            if 'anomaly_types' in anomalous_df.columns:
                anomaly_types = []
                for types in anomalous_df['anomaly_types']:
                    if isinstance(types, list):
                        anomaly_types.extend(types)
                    elif isinstance(types, str):
                        try:
                            import ast
                            parsed_types = ast.literal_eval(types)
                            if isinstance(parsed_types, list):
                                anomaly_types.extend(parsed_types)
                            else:
                                anomaly_types.append(types)
                        except:
                            anomaly_types.append(types)
                    else:
                        anomaly_types.append(str(types))
                
                if anomaly_types:
                    type_counts = pd.Series(anomaly_types).value_counts()
                    type_counts.plot(kind='bar', ax=ax1, alpha=0.7)
                    ax1.set_title('Anomaly Distribution by Type')
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                else:
                    ax1.text(0.5, 0.5, 'No valid anomaly type data', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Anomaly Distribution by Type')
            else:
                ax1.text(0.5, 0.5, 'No anomaly type data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Anomaly Distribution by Type')
            
            ax2 = axes[1]
            if 'anomaly_severity' in anomalous_df.columns:
                severities = anomalous_df['anomaly_severity']
                if len(severities) > 0:
                    severity_counts = severities.value_counts()
                    colors = [self.severity_colors.get(sev, 'gray') for sev in severity_counts.index]
                    severity_counts.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
                    ax2.set_title('Anomaly Distribution by Severity')
                    ax2.set_ylabel('Count')
                    ax2.tick_params(axis='x', rotation=45)
                else:
                    ax2.text(0.5, 0.5, 'No valid severity data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Anomaly Distribution by Severity')
            else:
                ax2.text(0.5, 0.5, 'No anomaly severity data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Anomaly Distribution by Severity')
            
            plt.tight_layout()
            filepath = self.output_dir / f"anomaly_distribution.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            logger.info("Saved anomaly distribution to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.error(f"Error in plot_anomaly_distribution: {e}")
            return ""

    def plot_feature_importance(self, features_df: pd.DataFrame, 
                              model_results: Optional[Dict[str, Any]] = None) -> str:
        try:
            logger.info("Plotting feature importance analysis")
            feature_columns = [col for col in features_df.columns 
                              if col not in ['is_anomalous', 'anomaly_severity', 'graph_id']]
            if not feature_columns:
                logger.warning("No features to analyze")
                return ""
            feature_data = features_df[feature_columns]
            feature_stats = []
            for col in feature_columns:
                if feature_data[col].dtype in ['int64', 'float64']:
                    stats = {
                        'feature': col,
                        'variance': feature_data[col].var(),
                        'mean': feature_data[col].mean(),
                        'unique_values': feature_data[col].nunique(),
                        'missing_rate': feature_data[col].isnull().sum() / len(feature_data)
                    }
                    feature_stats.append(stats)
            if not feature_stats:
                logger.warning("No numeric features to analyze")
                return ""
            stats_df = pd.DataFrame(feature_stats)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            ax1 = axes[0, 0]
            top_variance = stats_df.nlargest(20, 'variance')
            ax1.barh(range(len(top_variance)), top_variance['variance'])
            ax1.set_yticks(range(len(top_variance)))
            ax1.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_variance['feature']])
            ax1.set_title('Top 20 Features by Variance')
            ax1.set_xlabel('Variance')
            ax2 = axes[0, 1]
            top_unique = stats_df.nlargest(20, 'unique_values')
            ax2.barh(range(len(top_unique)), top_unique['unique_values'])
            ax2.set_yticks(range(len(top_unique)))
            ax2.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_unique['feature']])
            ax2.set_title('Top 20 Features by Unique Values')
            ax2.set_xlabel('Unique Values')
            ax3 = axes[1, 0]
            missing_features = stats_df[stats_df['missing_rate'] > 0].nlargest(20, 'missing_rate')
            if len(missing_features) > 0:
                ax3.barh(range(len(missing_features)), missing_features['missing_rate'])
                ax3.set_yticks(range(len(missing_features)))
                ax3.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in missing_features['feature']])
                ax3.set_title('Features with Missing Data')
                ax3.set_xlabel('Missing Rate')
            else:
                ax3.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Missing Data Analysis')
            ax4 = axes[1, 1]
            top_features = top_variance['feature'].head(10).tolist()
            if len(top_features) > 1:
                corr_data = feature_data[top_features].corr()
                im = ax4.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(top_features)))
                ax4.set_yticks(range(len(top_features)))
                ax4.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in top_features], rotation=45)
                ax4.set_yticklabels([f[:10] + '...' if len(f) > 10 else f for f in top_features])
                ax4.set_title('Feature Correlation Heatmap')
                plt.colorbar(im, ax=ax4)
            else:
                ax4.text(0.5, 0.5, 'Insufficient Features\nfor Correlation', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Correlation')
            plt.tight_layout()
            filepath = self.output_dir / f"feature_importance.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            logger.info("Saved feature importance analysis to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.error(f"Error in plot_feature_importance: {e}")
            return ""

    def plot_synthetic_data_quality(self, normal_trajectories: List, 
                                  anomalous_trajectories: List) -> str:
        """
        Analyze and visualize the quality of synthetic data generation.
        
        Args:
            normal_trajectories: List of normal trajectories
            anomalous_trajectories: List of anomalous trajectories
        
        Returns:
            Path to saved visualization
        """
        try:
            logger.info("Creating synthetic data quality analysis")
            
            # Extract metrics from trajectories
            normal_metrics = self._extract_trajectory_metrics(normal_trajectories)
            anomalous_metrics = self._extract_trajectory_metrics(anomalous_trajectories)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            # 1. Trajectory length distribution
            ax1 = axes[0, 0]
            normal_lengths = [t.total_duration for t in normal_trajectories]
            anomaly_lengths = [t.total_duration for t in anomalous_trajectories]
            
            ax1.hist(normal_lengths, bins=20, alpha=0.7, label='Normal', color='blue', density=True)
            ax1.hist(anomaly_lengths, bins=20, alpha=0.7, label='Anomalous', color='red', density=True)
            ax1.set_title('Trajectory Duration Distribution')
            ax1.set_xlabel('Duration (seconds)')
            ax1.set_ylabel('Density')
            ax1.legend()
            
            # 2. Node count distribution
            ax2 = axes[0, 1]
            normal_nodes = [len(t.nodes) for t in normal_trajectories]
            anomaly_nodes = [len(t.nodes) for t in anomalous_trajectories]
            
            ax2.hist(normal_nodes, bins=15, alpha=0.7, label='Normal', color='blue', density=True)
            ax2.hist(anomaly_nodes, bins=15, alpha=0.7, label='Anomalous', color='red', density=True)
            ax2.set_title('Node Count Distribution')
            ax2.set_xlabel('Number of Nodes')
            ax2.set_ylabel('Density')
            ax2.legend()
            
            # 3. Edge count distribution
            ax3 = axes[0, 2]
            normal_edges = [len(t.edges) for t in normal_trajectories]
            anomaly_edges = [len(t.edges) for t in anomalous_trajectories]
            
            ax3.hist(normal_edges, bins=15, alpha=0.7, label='Normal', color='blue', density=True)
            ax3.hist(anomaly_edges, bins=15, alpha=0.7, label='Anomalous', color='red', density=True)
            ax3.set_title('Edge Count Distribution')
            ax3.set_xlabel('Number of Edges')
            ax3.set_ylabel('Density')
            ax3.legend()
            
            # 4. Completion rate comparison
            ax4 = axes[1, 0]
            normal_completion = [t.completion_rate for t in normal_trajectories]
            anomaly_completion = [t.completion_rate for t in anomalous_trajectories]
            
            box_data = [normal_completion, anomaly_completion]
            box_labels = ['Normal', 'Anomalous']
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax4.set_title('Completion Rate Comparison')
            ax4.set_ylabel('Completion Rate')
            
            # 5. Success rate comparison
            ax5 = axes[1, 1]
            normal_success = [1 if t.success else 0 for t in normal_trajectories]
            anomaly_success = [1 if t.success else 0 for t in anomalous_trajectories]
            
            success_data = [normal_success, anomaly_success]
            bp2 = ax5.boxplot(success_data, labels=box_labels, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightblue')
            bp2['boxes'][1].set_facecolor('lightcoral')
            ax5.set_title('Success Rate Comparison')
            ax5.set_ylabel('Success (1) / Failure (0)')
            
            # 6. Node type distribution
            ax6 = axes[1, 2]
            normal_node_types = self._get_node_type_distribution(normal_trajectories)
            anomaly_node_types = self._get_node_type_distribution(anomalous_trajectories)
            
            # Combine and plot
            all_types = set(normal_node_types.keys()) | set(anomaly_node_types.keys())
            normal_counts = [normal_node_types.get(t, 0) for t in all_types]
            anomaly_counts = [anomaly_node_types.get(t, 0) for t in all_types]
            
            x = np.arange(len(all_types))
            width = 0.35
            
            ax6.bar(x - width/2, normal_counts, width, label='Normal', alpha=0.7, color='blue')
            ax6.bar(x + width/2, anomaly_counts, width, label='Anomalous', alpha=0.7, color='red')
            ax6.set_title('Node Type Distribution')
            ax6.set_xlabel('Node Types')
            ax6.set_ylabel('Count')
            ax6.set_xticks(x)
            ax6.set_xticklabels([t[:10] + '...' if len(t) > 10 else t for t in all_types], rotation=45)
            ax6.legend()
            
            # 7. Anomaly type distribution
            ax7 = axes[2, 0]
            anomaly_types = self._get_anomaly_type_distribution(anomalous_trajectories)
            if anomaly_types:
                types, counts = zip(*anomaly_types.items())
                ax7.bar(range(len(types)), counts, alpha=0.7, color='red')
                ax7.set_title('Anomaly Type Distribution')
                ax7.set_xlabel('Anomaly Types')
                ax7.set_ylabel('Count')
                ax7.set_xticks(range(len(types)))
                ax7.set_xticklabels([t[:15] + '...' if len(t) > 15 else t for t in types], rotation=45)
            else:
                ax7.text(0.5, 0.5, 'No anomaly type data', ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Anomaly Type Distribution')
            
            # 8. Anomaly severity distribution
            ax8 = axes[2, 1]
            severity_counts = self._get_severity_distribution(anomalous_trajectories)
            if severity_counts:
                severities, counts = zip(*severity_counts.items())
                colors = [self.severity_colors.get(sev, 'gray') for sev in severities]
                ax8.bar(range(len(severities)), counts, color=colors, alpha=0.7)
                ax8.set_title('Anomaly Severity Distribution')
                ax8.set_xlabel('Severity Levels')
                ax8.set_ylabel('Count')
                ax8.set_xticks(range(len(severities)))
                ax8.set_xticklabels(severities)
            else:
                ax8.text(0.5, 0.5, 'No severity data', ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Anomaly Severity Distribution')
            
            # 9. Data quality summary
            ax9 = axes[2, 2]
            summary_text = f"""
            Synthetic Data Quality Summary:
            
            Normal Trajectories: {len(normal_trajectories)}
            Anomalous Trajectories: {len(anomalous_trajectories)}
            
            Normal Metrics:
            - Avg Duration: {np.mean(normal_lengths):.1f}s
            - Avg Nodes: {np.mean(normal_nodes):.1f}
            - Avg Edges: {np.mean(normal_edges):.1f}
            - Success Rate: {np.mean(normal_success):.1%}
            
            Anomaly Metrics:
            - Avg Duration: {np.mean(anomaly_lengths):.1f}s
            - Avg Nodes: {np.mean(anomaly_nodes):.1f}
            - Avg Edges: {np.mean(anomaly_edges):.1f}
            - Success Rate: {np.mean(anomaly_success):.1%}
            
            Separation Quality:
            - Duration Diff: {abs(np.mean(normal_lengths) - np.mean(anomaly_lengths)):.1f}s
            - Node Count Diff: {abs(np.mean(normal_nodes) - np.mean(anomaly_nodes)):.1f}
            """
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax9.set_title('Quality Assessment')
            ax9.axis('off')
            
            plt.tight_layout()
            filepath = self.output_dir / f"synthetic_data_quality.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved synthetic data quality analysis to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_synthetic_data_quality: {e}")
            return ""

    def plot_anomaly_pattern_analysis(self, anomalous_trajectories: List) -> str:
        """
        Analyze patterns in the generated anomalies.
        
        Args:
            anomalous_trajectories: List of anomalous trajectories
        
        Returns:
            Path to saved visualization
        """
        try:
            logger.info("Creating anomaly pattern analysis")
            
            if not anomalous_trajectories:
                logger.warning("No anomalous trajectories to analyze")
                return ""
            
            # Extract pattern information
            pattern_data = self._extract_anomaly_patterns(anomalous_trajectories)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Anomaly type frequency
            ax1 = axes[0, 0]
            anomaly_types = pattern_data['anomaly_types']
            if anomaly_types:
                types, counts = zip(*anomaly_types.items())
                bars = ax1.bar(range(len(types)), counts, alpha=0.7, color='red')
                ax1.set_title('Anomaly Type Frequency')
                ax1.set_xlabel('Anomaly Types')
                ax1.set_ylabel('Count')
                ax1.set_xticks(range(len(types)))
                ax1.set_xticklabels([t[:12] + '...' if len(t) > 12 else t for t in types], rotation=45)
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom')
            
            # 2. Severity distribution
            ax2 = axes[0, 1]
            severity_data = pattern_data['severity_distribution']
            if severity_data:
                severities, counts = zip(*severity_data.items())
                colors = [self.severity_colors.get(sev, 'gray') for sev in severities]
                ax2.pie(counts, labels=severities, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Anomaly Severity Distribution')
            
            # 3. Anomaly patterns over trajectory length
            ax3 = axes[0, 2]
            length_patterns = pattern_data['length_patterns']
            if length_patterns:
                lengths = list(length_patterns.keys())
                anomaly_rates = list(length_patterns.values())
                ax3.scatter(lengths, anomaly_rates, alpha=0.7, color='red', s=50)
                ax3.set_title('Anomaly Rate vs Trajectory Length')
                ax3.set_xlabel('Trajectory Length (nodes)')
                ax3.set_ylabel('Anomaly Rate')
                ax3.grid(True, alpha=0.3)
            
            # 4. Node type anomaly susceptibility
            ax4 = axes[1, 0]
            node_susceptibility = pattern_data['node_susceptibility']
            if node_susceptibility:
                node_types, rates = zip(*node_susceptibility.items())
                bars = ax4.barh(range(len(node_types)), rates, alpha=0.7, color='orange')
                ax4.set_title('Node Type Anomaly Susceptibility')
                ax4.set_xlabel('Anomaly Rate')
                ax4.set_yticks(range(len(node_types)))
                ax4.set_yticklabels([t[:15] + '...' if len(t) > 15 else t for t in node_types])
                ax4.grid(True, alpha=0.3, axis='x')
            
            # 5. Temporal anomaly patterns
            ax5 = axes[1, 1]
            temporal_patterns = pattern_data['temporal_patterns']
            if temporal_patterns:
                positions = list(temporal_patterns.keys())
                anomaly_counts = list(temporal_patterns.values())
                ax5.plot(positions, anomaly_counts, marker='o', linewidth=2, markersize=6, color='red')
                ax5.set_title('Anomaly Occurrence by Position')
                ax5.set_xlabel('Node Position in Trajectory')
                ax5.set_ylabel('Anomaly Count')
                ax5.grid(True, alpha=0.3)
            
            # 6. Anomaly complexity analysis
            ax6 = axes[1, 2]
            complexity_data = pattern_data['complexity_analysis']
            if complexity_data:
                # Create a scatter plot of complexity vs anomaly count
                complexities = list(complexity_data.keys())
                anomaly_counts = list(complexity_data.values())
                ax6.scatter(complexities, anomaly_counts, alpha=0.7, color='purple', s=60)
                ax6.set_title('Anomaly Complexity Analysis')
                ax6.set_xlabel('Trajectory Complexity (edges/nodes)')
                ax6.set_ylabel('Anomaly Count')
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = self.output_dir / f"anomaly_pattern_analysis.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved anomaly pattern analysis to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_anomaly_pattern_analysis: {e}")
            return ""

    def _extract_trajectory_metrics(self, trajectories: List) -> Dict[str, List]:
        """Extract metrics from trajectories for analysis."""
        metrics = {
            'durations': [],
            'node_counts': [],
            'edge_counts': [],
            'completion_rates': [],
            'success_rates': []
        }
        
        for trajectory in trajectories:
            metrics['durations'].append(getattr(trajectory, 'total_duration', 0))
            metrics['node_counts'].append(len(trajectory.nodes))
            metrics['edge_counts'].append(len(trajectory.edges))
            metrics['completion_rates'].append(getattr(trajectory, 'completion_rate', 0))
            metrics['success_rates'].append(1 if getattr(trajectory, 'success', False) else 0)
        
        return metrics

    def _get_node_type_distribution(self, trajectories: List) -> Dict[str, int]:
        """Get distribution of node types across trajectories."""
        node_types = {}
        for trajectory in trajectories:
            for node in trajectory.nodes:
                node_type = node.node_type.value
                node_types[node_type] = node_types.get(node_type, 0) + 1
        return node_types

    def _get_anomaly_type_distribution(self, trajectories: List) -> Dict[str, int]:
        """Get distribution of anomaly types."""
        anomaly_types = {}
        for trajectory in trajectories:
            if hasattr(trajectory, 'anomaly_types') and trajectory.anomaly_types:
                for anomaly_type in trajectory.anomaly_types:
                    # Handle both string and enum types
                    if hasattr(anomaly_type, 'value'):
                        anomaly_type_str = anomaly_type.value
                    else:
                        anomaly_type_str = str(anomaly_type)
                    anomaly_types[anomaly_type_str] = anomaly_types.get(anomaly_type_str, 0) + 1
        return dict(sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True))

    def _get_severity_distribution(self, trajectories: List) -> Dict[str, int]:
        """Get distribution of anomaly severities."""
        severities = {}
        for trajectory in trajectories:
            if hasattr(trajectory, 'anomaly_severity') and trajectory.anomaly_severity:
                severity = trajectory.anomaly_severity
                severities[severity] = severities.get(severity, 0) + 1
        return severities

    def _extract_anomaly_patterns(self, trajectories: List) -> Dict[str, Any]:
        """Extract comprehensive anomaly patterns from trajectories."""
        patterns = {
            'anomaly_types': {},
            'severity_distribution': {},
            'length_patterns': {},
            'node_susceptibility': {},
            'temporal_patterns': {},
            'complexity_analysis': {}
        }
        
        # Analyze anomaly types
        for trajectory in trajectories:
            if hasattr(trajectory, 'anomaly_types') and trajectory.anomaly_types:
                for anomaly_type in trajectory.anomaly_types:
                    patterns['anomaly_types'][anomaly_type] = patterns['anomaly_types'].get(anomaly_type, 0) + 1
        
        # Sort by frequency
        patterns['anomaly_types'] = dict(
            sorted(patterns['anomaly_types'].items(), key=lambda x: x[1], reverse=True)
        )
        
        # Analyze severity distribution
        for trajectory in trajectories:
            if hasattr(trajectory, 'anomaly_severity') and trajectory.anomaly_severity:
                severity = trajectory.anomaly_severity
                patterns['severity_distribution'][severity] = patterns['severity_distribution'].get(severity, 0) + 1
        
        # Analyze length patterns
        length_groups = {}
        for trajectory in trajectories:
            length = len(trajectory.nodes)
            if length not in length_groups:
                length_groups[length] = 0
            length_groups[length] += 1
        
        if length_groups:
            patterns['length_patterns'] = length_groups
        
        # Analyze node type susceptibility
        node_anomaly_counts = {}
        total_node_counts = {}
        
        for trajectory in trajectories:
            for node in trajectory.nodes:
                node_type = node.node_type.value
                total_node_counts[node_type] = total_node_counts.get(node_type, 0) + 1
                if node.is_anomalous:
                    node_anomaly_counts[node_type] = node_anomaly_counts.get(node_type, 0) + 1
        
        for node_type in total_node_counts:
            if node_type in node_anomaly_counts:
                susceptibility = node_anomaly_counts[node_type] / total_node_counts[node_type]
                patterns['node_susceptibility'][node_type] = susceptibility
        
        # Sort by susceptibility
        patterns['node_susceptibility'] = dict(
            sorted(patterns['node_susceptibility'].items(), 
                  key=lambda x: x[1], reverse=True)
        )
        
        # Analyze temporal patterns
        temporal_counts = {}
        for trajectory in trajectories:
            for i, node in enumerate(trajectory.nodes):
                if node.is_anomalous:
                    temporal_counts[i] = temporal_counts.get(i, 0) + 1
        
        if temporal_counts:
            patterns['temporal_patterns'] = temporal_counts
        
        # Analyze complexity patterns
        complexity_counts = {}
        for trajectory in trajectories:
            complexity = len(trajectory.edges) / max(len(trajectory.nodes), 1)
            complexity_rounded = round(complexity, 1)
            complexity_counts[complexity_rounded] = complexity_counts.get(complexity_rounded, 0) + 1
        
        if complexity_counts:
            patterns['complexity_analysis'] = complexity_counts
        
        return patterns