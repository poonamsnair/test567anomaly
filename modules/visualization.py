"""
Visualization system for AI Agent Trajectory Anomaly Detection.

This module creates comprehensive visualizations including:
- Agent trajectory DAG plots
- Model performance comparisons
- Embedding space visualizations
- Anomaly detection results
- Feature importance analysis
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Some visualizations will be limited.")

from .utils import ensure_directory

# Suppress warnings for cleaner output
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


class AnomalyDetectionVisualizer:
    """
    Comprehensive visualization system for anomaly detection results.
    
    This class creates various types of visualizations:
    1. Trajectory graph visualizations
    2. Model performance comparisons
    3. Embedding space plots
    4. Anomaly analysis charts
    5. Feature importance plots
    """
    
    def __init__(self, config: Dict, output_dir: str = "charts"):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save charts
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
        
        # Setup matplotlib
        setup_matplotlib_for_plotting()
        
        # Color schemes
        self.color_schemes = self.viz_config.get('color_schemes', {})
        self.node_colors = {
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
        
        logger.info("AnomalyDetectionVisualizer initialized, output dir: %s", self.output_dir)
    
    def plot_trajectory_examples(self, graphs: List[nx.DiGraph], 
                                normal_indices: List[int], anomalous_indices: List[int],
                                max_examples: int = 6) -> str:
        """
        Plot example trajectories showing normal vs anomalous patterns.
        
        Args:
            graphs: List of trajectory graphs
            normal_indices: Indices of normal trajectories
            anomalous_indices: Indices of anomalous trajectories
            max_examples: Maximum number of examples to show
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting trajectory examples")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select examples
        normal_examples = normal_indices[:max_examples//2]
        anomalous_examples = anomalous_indices[:max_examples//2]
        
        if not normal_examples and not anomalous_examples:
            logger.warning("No examples to plot")
            return ""
        
        # Create subplot grid
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
        
        # Plot normal examples
        for i, idx in enumerate(normal_examples):
            if idx < len(graphs):
                self._plot_single_trajectory(graphs[idx], axes[i], f"Normal {idx}")
        
        # Plot anomalous examples
        for i, idx in enumerate(anomalous_examples):
            plot_idx = len(normal_examples) + i
            if idx < len(graphs):
                self._plot_single_trajectory(graphs[idx], axes[plot_idx], f"Anomalous {idx}")
        
        # Hide unused subplots
        for i in range(total_examples, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.output_dir / f"trajectory_examples.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved trajectory examples to %s", filepath)
        return str(filepath)
    
    def _plot_single_trajectory(self, graph: nx.DiGraph, ax, title: str) -> None:
        """Plot a single trajectory graph."""
        if len(graph.nodes) == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Choose layout based on graph structure
        if len(graph.nodes) < 20:
            try:
                pos = nx.spring_layout(graph, k=1, iterations=50)
            except:
                pos = nx.random_layout(graph)
        else:
            pos = nx.random_layout(graph)
        
        # Draw nodes colored by type
        for node_type in self.node_colors:
            node_list = [node for node, data in graph.nodes(data=True) 
                        if data.get('node_type') == node_type]
            if node_list:
                nx.draw_networkx_nodes(
                    graph, pos, nodelist=node_list,
                    node_color=self.node_colors[node_type],
                    node_size=200, alpha=0.8, ax=ax
                )
        
        # Draw other nodes in default color
        other_nodes = [node for node, data in graph.nodes(data=True)
                      if data.get('node_type') not in self.node_colors]
        if other_nodes:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=other_nodes,
                node_color='lightgray', node_size=200, alpha=0.8, ax=ax
            )
        
        # Draw edges
        normal_edges = [(u, v) for u, v, data in graph.edges(data=True)
                       if not data.get('is_anomalous', False)]
        anomalous_edges = [(u, v) for u, v, data in graph.edges(data=True)
                          if data.get('is_anomalous', False)]
        
        if normal_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=normal_edges,
                edge_color='gray', alpha=0.6, arrows=True, ax=ax
            )
        
        if anomalous_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=anomalous_edges,
                edge_color='red', alpha=0.8, arrows=True, width=2, ax=ax
            )
        
        ax.set_title(title)
        ax.axis('off')
    
    def plot_model_performance_comparison(self, model_results: Dict[str, Any]) -> str:
        """
        Plot model performance comparison.
        
        Args:
            model_results: Dictionary of model results
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting model performance comparison")
        
        # Extract metrics
        models = []
        metrics_data = []
        
        for model_name, results in model_results.items():
            if 'test_metrics' in results:
                models.append(model_name)
                metrics = results['test_metrics']
                metrics_data.append({
                    'Model': model_name,
                    'F1 Score': metrics.get('f1', 0.0),
                    'Precision': metrics.get('precision', 0.0),
                    'Recall': metrics.get('recall', 0.0),
                    'AUC-ROC': metrics.get('auc_roc', 0.5),
                    'AUC-PR': metrics.get('auc_pr', 0.0)
                })
        
        if not metrics_data:
            logger.warning("No model metrics to plot")
            return ""
        
        df = pd.DataFrame(metrics_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC', 'AUC-PR']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            bars = ax.bar(df['Model'], df[metric], alpha=0.7)
            
            # Color bars based on performance
            for j, bar in enumerate(bars):
                value = df[metric].iloc[j]
                if value >= 0.8:
                    bar.set_color('green')
                elif value >= 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.output_dir / f"model_performance_comparison.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved model performance comparison to %s", filepath)
        return str(filepath)
    
    def plot_roc_curves(self, model_results: Dict[str, Any], 
                       test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot ROC curves for all models.
        
        Args:
            model_results: Dictionary of model results
            test_df: Test DataFrame with labels
            test_features: Test features
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting ROC curves")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Get true labels
        y_true = test_df.get('is_anomalous', pd.Series([False] * len(test_df))).astype(int).values
        
        if len(set(y_true)) < 2:
            logger.warning("Cannot plot ROC curves - only one class present")
            return ""
        
        try:
            from .models import AnomalyDetectionModels
            models_handler = AnomalyDetectionModels(self.config)
            
            for model_name, model_data in model_results.items():
                if model_data.get('model') is not None:
                    try:
                        # Use the same feature columns as training
                        feature_columns = model_data.get('feature_names')
                        if feature_columns:
                            # Prepare test data with same features as training
                            test_features_prepared, _ = models_handler.prepare_training_data(
                                test_df, feature_columns=feature_columns
                            )
                        else:
                            test_features_prepared = test_features
                        
                        scores, _ = models_handler.predict_anomalies(model_data, test_features_prepared)
                        
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                        
                    except Exception as e:
                        logger.warning("Failed to plot ROC curve for %s: %s", model_name, e)
                        continue
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save figure
            filepath = self.output_dir / f"roc_curves.{self.format}"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved ROC curves to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error("Failed to plot ROC curves: %s", e)
            plt.close()
            return ""
    
    def plot_precision_recall_curves(self, model_results: Dict[str, Any], 
                                   test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            model_results: Dictionary of model results
            test_df: Test DataFrame with labels
            test_features: Test features
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting Precision-Recall curves")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Get true labels
        y_true = test_df.get('is_anomalous', pd.Series([False] * len(test_df))).astype(int).values
        
        if len(set(y_true)) < 2 or y_true.sum() == 0:
            logger.warning("Cannot plot PR curves - no positive examples")
            return ""
        
        try:
            from .models import AnomalyDetectionModels
            models_handler = AnomalyDetectionModels(self.config)
            
            for model_name, model_data in model_results.items():
                if model_data.get('model') is not None:
                    try:
                        # Use the same feature columns as training
                        feature_columns = model_data.get('feature_names')
                        if feature_columns:
                            # Prepare test data with same features as training
                            test_features_prepared, _ = models_handler.prepare_training_data(
                                test_df, feature_columns=feature_columns
                            )
                        else:
                            test_features_prepared = test_features
                        
                        scores, _ = models_handler.predict_anomalies(model_data, test_features_prepared)
                        
                        from sklearn.metrics import precision_recall_curve, auc
                        precision, recall, _ = precision_recall_curve(y_true, scores)
                        pr_auc = auc(recall, precision)
                        
                        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
                        
                    except Exception as e:
                        logger.warning("Failed to plot PR curve for %s: %s", model_name, e)
                        continue
            
            # Plot baseline
            baseline = y_true.sum() / len(y_true)
            plt.axhline(y=baseline, color='navy', linestyle='--', alpha=0.8, 
                       label=f'Baseline (Random = {baseline:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            # Save figure
            filepath = self.output_dir / f"precision_recall_curves.{self.format}"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved Precision-Recall curves to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error("Failed to plot PR curves: %s", e)
            plt.close()
            return ""
    
    def plot_embedding_visualization(self, embeddings: Dict[str, np.ndarray], 
                                   labels: np.ndarray, method: str = 'tsne') -> str:
        """
        Plot embedding space visualization.
        
        Args:
            embeddings: Dictionary of embeddings
            labels: True labels for coloring
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting embedding visualization using %s", method)
        
        if not embeddings:
            logger.warning("No embeddings to visualize")
            return ""
        
        # Convert embeddings to matrix
        embedding_matrix = np.array(list(embeddings.values()))
        
        if embedding_matrix.shape[1] < 2:
            logger.warning("Embeddings have insufficient dimensions for visualization")
            return ""
        
        # Reduce dimensionality
        try:
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding_matrix)//4))
                embedding_2d = reducer.fit_transform(embedding_matrix)
            elif method == 'umap' and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embedding_matrix)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embedding_matrix)
            else:
                # Fallback to PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embedding_matrix)
                method = 'pca'
        except Exception as e:
            logger.warning("Failed to reduce dimensions: %s", e)
            return ""
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color points by anomaly status
        normal_mask = labels == 0
        anomalous_mask = labels == 1
        
        if normal_mask.sum() > 0:
            plt.scatter(embedding_2d[normal_mask, 0], embedding_2d[normal_mask, 1], 
                       c='blue', alpha=0.6, s=50, label='Normal')
        
        if anomalous_mask.sum() > 0:
            plt.scatter(embedding_2d[anomalous_mask, 0], embedding_2d[anomalous_mask, 1], 
                       c='red', alpha=0.8, s=50, label='Anomalous')
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Embedding Space Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        filepath = self.output_dir / f"embedding_visualization_{method}.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved embedding visualization to %s", filepath)
        return str(filepath)
    
    def plot_anomaly_distribution(self, test_df: pd.DataFrame) -> str:
        """
        Plot distribution of anomalies by type and severity.
        
        Args:
            test_df: Test DataFrame with anomaly information
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting anomaly distribution")
        
        anomalous_df = test_df[test_df.get('is_anomalous', False)]
        
        if len(anomalous_df) == 0:
            logger.warning("No anomalous trajectories to analyze")
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot by anomaly type
        ax1 = axes[0]
        anomaly_types = []
        
        for _, row in anomalous_df.iterrows():
            types = row.get('anomaly_types', [])
            if isinstance(types, str):
                anomaly_types.append(types)
            elif isinstance(types, list):
                anomaly_types.extend(types)
        
        if anomaly_types:
            type_counts = pd.Series(anomaly_types).value_counts()
            type_counts.plot(kind='bar', ax=ax1, alpha=0.7)
            ax1.set_title('Anomaly Distribution by Type')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot by severity
        ax2 = axes[1]
        severities = anomalous_df.get('anomaly_severity', pd.Series(['unknown'] * len(anomalous_df)))
        severity_counts = severities.value_counts()
        
        # Use severity colors
        colors = [self.severity_colors.get(sev, 'gray') for sev in severity_counts.index]
        severity_counts.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
        ax2.set_title('Anomaly Distribution by Severity')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.output_dir / f"anomaly_distribution.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved anomaly distribution to %s", filepath)
        return str(filepath)
    
    def plot_feature_importance(self, features_df: pd.DataFrame, 
                              model_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Plot feature importance analysis.
        
        Args:
            features_df: Feature DataFrame
            model_results: Optional model results for feature importance
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting feature importance analysis")
        
        # Remove label columns for analysis
        feature_columns = [col for col in features_df.columns 
                          if col not in ['is_anomalous', 'anomaly_severity', 'graph_id']]
        
        if not feature_columns:
            logger.warning("No features to analyze")
            return ""
        
        feature_data = features_df[feature_columns]
        
        # Calculate feature statistics
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
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature variance
        ax1 = axes[0, 0]
        top_variance = stats_df.nlargest(20, 'variance')
        ax1.barh(range(len(top_variance)), top_variance['variance'])
        ax1.set_yticks(range(len(top_variance)))
        ax1.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_variance['feature']])
        ax1.set_title('Top 20 Features by Variance')
        ax1.set_xlabel('Variance')
        
        # Feature uniqueness
        ax2 = axes[0, 1]
        top_unique = stats_df.nlargest(20, 'unique_values')
        ax2.barh(range(len(top_unique)), top_unique['unique_values'])
        ax2.set_yticks(range(len(top_unique)))
        ax2.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_unique['feature']])
        ax2.set_title('Top 20 Features by Unique Values')
        ax2.set_xlabel('Unique Values')
        
        # Missing data
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
        
        # Feature correlation heatmap (top features)
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
        
        # Save figure
        filepath = self.output_dir / f"feature_importance.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved feature importance analysis to %s", filepath)
        return str(filepath)
    
    def plot_threshold_calibration(self, threshold_results: Dict[str, Dict[str, float]]) -> str:
        """
        Plot threshold calibration results.
        
        Args:
            threshold_results: Dictionary of threshold calibration results
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting threshold calibration results")
        
        if not threshold_results:
            logger.warning("No threshold results to plot")
            return ""
        
        # Extract data
        methods = list(threshold_results.keys())
        metrics = ['f1', 'precision', 'recall']
        
        data = []
        for method in methods:
            for metric in metrics:
                value = threshold_results[method].get(metric, 0.0)
                data.append({
                    'Method': method,
                    'Metric': metric.capitalize(),
                    'Score': value
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 8))
        
        # Pivot for easier plotting
        pivot_df = df.pivot(index='Method', columns='Metric', values='Score')
        
        ax = pivot_df.plot(kind='bar', alpha=0.7, width=0.8)
        ax.set_title('Threshold Calibration Results')
        ax.set_ylabel('Score')
        ax.set_xlabel('Calibration Method')
        ax.legend(title='Metric')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.output_dir / f"threshold_calibration.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved threshold calibration results to %s", filepath)
        return str(filepath)
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> str:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            results: Complete results dictionary
        
        Returns:
            Path to saved HTML file
        """
        logger.info("Creating interactive dashboard")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance', 'Anomaly Distribution', 
                              'Feature Importance', 'Threshold Analysis'),
                specs=[[{'type': 'bar'}, {'type': 'pie'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Model performance
            if 'model_results' in results:
                models = []
                f1_scores = []
                
                for model_name, model_data in results['model_results'].items():
                    if 'test_metrics' in model_data:
                        models.append(model_name)
                        f1_scores.append(model_data['test_metrics'].get('f1', 0.0))
                
                if models:
                    fig.add_trace(
                        go.Bar(x=models, y=f1_scores, name='F1 Score'),
                        row=1, col=1
                    )
            
            # Anomaly distribution (placeholder)
            anomaly_types = ['infinite_loops', 'suboptimal_paths', 'tool_failures', 'others']
            counts = [25, 20, 15, 10]  # Placeholder data
            
            fig.add_trace(
                go.Pie(labels=anomaly_types, values=counts, name='Anomaly Types'),
                row=1, col=2
            )
            
            # Feature importance (placeholder)
            features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
            importance = [0.8, 0.6, 0.4, 0.3, 0.2]
            
            fig.add_trace(
                go.Bar(x=features, y=importance, name='Importance'),
                row=2, col=1
            )
            
            # Threshold analysis
            if 'threshold_results' in results:
                methods = list(results['threshold_results'].keys())
                f1_values = [results['threshold_results'][method].get('f1', 0.0) for method in methods]
                
                fig.add_trace(
                    go.Bar(x=methods, y=f1_values, name='F1 Score'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Anomaly Detection Analysis Dashboard"
            )
            
            # Save as HTML
            filepath = self.output_dir / "interactive_dashboard.html"
            fig.write_html(str(filepath))
            
            logger.info("Saved interactive dashboard to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error("Failed to create interactive dashboard: %s", e)
            return ""
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a visual summary report.
        
        Args:
            results: Complete results dictionary
        
        Returns:
            Path to saved figure
        """
        logger.info("Generating summary report")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Summary statistics
        ax1 = fig.add_subplot(gs[0, :])
        summary_text = self._create_summary_text(results)
        ax1.text(0.05, 0.5, summary_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.set_title('Analysis Summary', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Model comparison
        ax2 = fig.add_subplot(gs[1, 0])
        if 'model_results' in results:
            models = []
            f1_scores = []
            
            for model_name, model_data in results['model_results'].items():
                if 'test_metrics' in model_data:
                    models.append(model_name)
                    f1_scores.append(model_data['test_metrics'].get('f1', 0.0))
            
            if models:
                bars = ax2.bar(models, f1_scores)
                ax2.set_title('Model F1 Scores')
                ax2.set_ylabel('F1 Score')
                ax2.set_ylim(0, 1)
                
                # Color bars
                for i, bar in enumerate(bars):
                    if f1_scores[i] >= 0.8:
                        bar.set_color('green')
                    elif f1_scores[i] >= 0.6:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
        
        # Recommendations
        ax3 = fig.add_subplot(gs[1:, 1:])
        recommendations = self._generate_recommendations(results)
        ax3.text(0.05, 0.95, recommendations, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax3.set_title('Recommendations', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle('Anomaly Detection System Analysis Report', fontsize=18, fontweight='bold')
        
        # Save figure
        filepath = self.output_dir / f"summary_report.{self.format}"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
        plt.close()
        
        logger.info("Saved summary report to %s", filepath)
        return str(filepath)
    
    def _create_summary_text(self, results: Dict[str, Any]) -> str:
        """Create summary text for the report."""
        lines = []
        
        # Dataset information
        total_trajectories = results.get('total_trajectories', 0)
        anomaly_rate = results.get('anomaly_rate', 0.0)
        lines.append(f"Dataset: {total_trajectories} trajectories ({anomaly_rate:.1%} anomalous)")
        
        # Best model
        if 'model_results' in results:
            best_model = None
            best_f1 = 0.0
            
            for model_name, model_data in results['model_results'].items():
                if 'test_metrics' in model_data:
                    f1 = model_data['test_metrics'].get('f1', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_name
            
            if best_model:
                lines.append(f"Best Model: {best_model} (F1: {best_f1:.3f})")
        
        # Training information
        train_time = results.get('training_time', 0.0)
        if train_time > 0:
            lines.append(f"Training Time: {train_time:.1f} seconds")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on results."""
        recommendations = ["RECOMMENDATIONS:\n"]
        
        # Model performance recommendations
        if 'model_results' in results:
            best_f1 = 0.0
            worst_f1 = 1.0
            
            for model_name, model_data in results['model_results'].items():
                if 'test_metrics' in model_data:
                    f1 = model_data['test_metrics'].get('f1', 0.0)
                    best_f1 = max(best_f1, f1)
                    worst_f1 = min(worst_f1, f1)
            
            if best_f1 < 0.7:
                recommendations.append("• Overall performance is low - consider feature engineering")
            
            if best_f1 - worst_f1 > 0.2:
                recommendations.append("• Large performance gap between models - ensemble might help")
        
        # Data recommendations
        anomaly_rate = results.get('anomaly_rate', 0.0)
        if anomaly_rate < 0.05:
            recommendations.append("• Very low anomaly rate - consider generating more anomalous examples")
        elif anomaly_rate > 0.3:
            recommendations.append("• High anomaly rate - verify data quality")
        
        # Feature recommendations
        feature_count = results.get('feature_count', 0)
        if feature_count > 1000:
            recommendations.append("• High-dimensional features - consider dimensionality reduction")
        elif feature_count < 10:
            recommendations.append("• Few features - consider feature engineering")
        
        # General recommendations
        recommendations.extend([
            "• Monitor model performance over time",
            "• Consider domain-specific feature engineering",
            "• Validate results with domain experts",
            "• Implement automated retraining pipeline"
        ])
        
        return "\n".join(recommendations)
