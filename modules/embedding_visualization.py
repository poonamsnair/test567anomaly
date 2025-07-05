"""
Embedding Visualization Module for AI Agent Trajectory Anomaly Detection.

This module provides comprehensive visualization capabilities for:
- UMAP and t-SNE dimensionality reduction plots
- Embedding method comparisons (Node2Vec, DeepWalk, GraphSAGE)
- Embedding quality analysis
- Confusion matrices for all models
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

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
    """Setup matplotlib and seaborn for plotting with proper configuration."""
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


class EmbeddingVisualizer:
    """
    Comprehensive embedding visualization system.
    
    This class creates various types of embedding visualizations:
    1. UMAP dimensionality reduction plots
    2. t-SNE dimensionality reduction plots
    3. Embedding method comparisons
    4. Confusion matrices for all models
    5. Embedding quality analysis
    """
    
    def __init__(self, config: Dict, output_dir: str = "charts"):
        """
        Initialize embedding visualizer.
        
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
        self.colors = {
            'normal': '#2E8B57',  # Sea Green
            'anomalous': '#DC143C',  # Crimson
            'node2vec': '#1f77b4',
            'deepwalk': '#ff7f0e', 
            'graphsage': '#2ca02c'
        }
        
        # Figure settings
        self.figure_settings = self.viz_config.get('figure_settings', {})
        self.dpi = self.figure_settings.get('dpi', 300)
        self.format = self.figure_settings.get('format', 'png')
        self.bbox_inches = self.figure_settings.get('bbox_inches', 'tight')
        
        logger.info("EmbeddingVisualizer initialized, output dir: %s", self.output_dir)
    
    def plot_umap_embeddings(self, embeddings: Dict[str, np.ndarray], 
                           labels: np.ndarray, method: str = 'umap') -> str:
        """
        Create UMAP dimensionality reduction plot for embeddings.
        
        Args:
            embeddings: Dictionary of graph embeddings
            labels: Binary labels (0=normal, 1=anomalous)
            method: Dimensionality reduction method ('umap' or 'tsne')
        
        Returns:
            Path to saved plot
        """
        try:
            logger.info(f"Creating {method.upper()} embedding visualization")
            
            if not UMAP_AVAILABLE and method == 'umap':
                logger.warning("UMAP not available, falling back to t-SNE")
                method = 'tsne'
            
            # Convert embeddings to matrix
            embedding_matrix = np.array(list(embeddings.values()))
            
            if embedding_matrix.shape[0] == 0:
                logger.warning("No embeddings to visualize")
                return ""
            
            # Standardize embeddings
            scaler = StandardScaler()
            embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)
            
            # Apply dimensionality reduction
            if method == 'umap':
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(embedding_matrix) - 1),
                    min_dist=0.1
                )
                coords = reducer.fit_transform(embedding_matrix_scaled)
                title = "UMAP Visualization of Graph Embeddings"
            else:  # t-SNE
                reducer = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embedding_matrix) - 1),
                    n_iter=1000
                )
                coords = reducer.fit_transform(embedding_matrix_scaled)
                title = "t-SNE Visualization of Graph Embeddings"
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Ensure labels are 1D
            if labels.ndim > 1:
                if labels.shape[1] == 2:  # One-hot encoded
                    labels_1d = labels[:, 1]  # Take the second column (anomalous class)
                else:
                    labels_1d = labels.flatten()
            else:
                labels_1d = labels
            
            # Plot points
            normal_mask = labels_1d == 0
            anomalous_mask = labels_1d == 1
            
            if np.any(normal_mask):
                ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1], 
                          c=self.colors['normal'], label='Normal', alpha=0.7, s=50)
            
            if np.any(anomalous_mask):
                ax.scatter(coords[anomalous_mask, 0], coords[anomalous_mask, 1], 
                          c=self.colors['anomalous'], label='Anomalous', alpha=0.7, s=50)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            normal_count = np.sum(normal_mask)
            anomalous_count = np.sum(anomalous_mask)
            total_count = len(labels)
            
            stats_text = f'Total: {total_count}\nNormal: {normal_count}\nAnomalous: {anomalous_count}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            filename = f"embedding_{method}_visualization.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved {method.upper()} visualization to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating {method.upper()} visualization: {e}")
            return ""
    
    def plot_tsne_embeddings(self, embeddings: Dict[str, np.ndarray], 
                           labels: np.ndarray) -> str:
        """
        Create t-SNE dimensionality reduction plot for embeddings.
        
        Args:
            embeddings: Dictionary of graph embeddings
            labels: Binary labels (0=normal, 1=anomalous)
        
        Returns:
            Path to saved plot
        """
        return self.plot_umap_embeddings(embeddings, labels, method='tsne')
    
    def plot_embedding_method_comparison(self, embeddings_dict: Dict[str, Dict[str, np.ndarray]], 
                                       labels: np.ndarray) -> str:
        """
        Create comparison plot for different embedding methods.
        
        Args:
            embeddings_dict: Dictionary of embedding results for different methods
            labels: Binary labels (0=normal, 1=anomalous)
        
        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating embedding method comparison visualization")
            
            # Filter out methods with errors
            valid_methods = {}
            for method, data in embeddings_dict.items():
                if 'error' not in data and 'embeddings' in data:
                    valid_methods[method] = data['embeddings']
            
            if len(valid_methods) < 2:
                logger.warning("Need at least 2 valid embedding methods for comparison")
                return ""
            
            # Create subplot grid
            n_methods = len(valid_methods)
            cols = min(3, n_methods)
            rows = (n_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_methods == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # Plot each method
            for i, (method, embeddings) in enumerate(valid_methods.items()):
                ax = axes[i]
                
                # Convert embeddings to matrix
                embedding_matrix = np.array(list(embeddings.values()))
                
                if embedding_matrix.shape[0] == 0:
                    ax.text(0.5, 0.5, f"No data for {method}", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{method.upper()} Embeddings", fontweight='bold')
                    continue
                
                # Standardize embeddings
                scaler = StandardScaler()
                embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)
                
                # Apply PCA for 2D visualization
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(embedding_matrix_scaled)
                
                # Ensure labels are 1D
                if labels.ndim > 1:
                    if labels.shape[1] == 2:  # One-hot encoded
                        labels_1d = labels[:, 1]  # Take the second column (anomalous class)
                    else:
                        labels_1d = labels.flatten()
                else:
                    labels_1d = labels
                
                # Plot points
                normal_mask = labels_1d == 0
                anomalous_mask = labels_1d == 1
                
                if np.any(normal_mask):
                    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1], 
                              c=self.colors['normal'], label='Normal', alpha=0.7, s=30)
                
                if np.any(anomalous_mask):
                    ax.scatter(coords[anomalous_mask, 0], coords[anomalous_mask, 1], 
                              c=self.colors['anomalous'], label='Anomalous', alpha=0.7, s=30)
                
                ax.set_title(f"{method.upper()} Embeddings", fontweight='bold')
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add explained variance
                explained_var = pca.explained_variance_ratio_
                ax.text(0.02, 0.98, f'Explained variance:\n{explained_var[0]:.2%}\n{explained_var[1]:.2%}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_methods, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"embedding_method_comparison.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved embedding method comparison to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating embedding method comparison: {e}")
            return ""
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Create confusion matrices for all models.
        
        Args:
            evaluation_results: Dictionary containing evaluation results with predictions
        
        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating confusion matrices for all models")
            
            # Get model predictions from evaluation results
            model_predictions = {}
            for model_name, eval_data in evaluation_results.items():
                if 'error' not in eval_data and 'predictions' in eval_data and 'y_true' in eval_data:
                    predictions = eval_data['predictions']
                    true_labels = eval_data['y_true']
                    
                    if len(predictions) > 0 and len(true_labels) > 0:
                        model_predictions[model_name] = {
                            'predictions': predictions,
                            'true_labels': true_labels
                        }
            
            if not model_predictions:
                logger.warning("No valid model predictions found")
                return ""
            
            # Create subplot grid
            n_models = len(model_predictions)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # Plot confusion matrix for each model
            for i, (model_name, pred_data) in enumerate(model_predictions.items()):
                ax = axes[i]
                
                # Calculate confusion matrix
                cm = confusion_matrix(pred_data['true_labels'], pred_data['predictions'])
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Normal', 'Anomalous'],
                           yticklabels=['Normal', 'Anomalous'])
                
                ax.set_title(f"{model_name.replace('_', ' ').title()} Confusion Matrix", fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                # Add metrics
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"confusion_matrices.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved confusion matrices to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating confusion matrices: {e}")
            return ""
    
    def plot_embedding_quality_analysis(self, embeddings_dict: Dict[str, Dict[str, np.ndarray]], 
                                      labels: np.ndarray) -> str:
        """
        Create comprehensive embedding quality analysis.
        
        Args:
            embeddings_dict: Dictionary of embedding results for different methods
            labels: Binary labels (0=normal, 1=anomalous)
        
        Returns:
            Path to saved plot
        """
        try:
            logger.info("Creating embedding quality analysis")
            
            # Filter out methods with errors
            valid_methods = {}
            for method, data in embeddings_dict.items():
                if 'error' not in data and 'embeddings' in data:
                    valid_methods[method] = data['embeddings']
            
            if not valid_methods:
                logger.warning("No valid embedding methods found")
                return ""
            
            # Calculate quality metrics for each method
            quality_metrics = {}
            for method, embeddings in valid_methods.items():
                embedding_matrix = np.array(list(embeddings.values()))
                
                if embedding_matrix.shape[0] == 0:
                    continue
                
                # Calculate various quality metrics
                metrics = self._calculate_embedding_quality_metrics(embedding_matrix, labels)
                quality_metrics[method] = metrics
            
            if not quality_metrics:
                logger.warning("No quality metrics calculated")
                return ""
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Silhouette Score Comparison
            methods = list(quality_metrics.keys())
            silhouette_scores = [quality_metrics[m]['silhouette_score'] for m in methods]
            
            ax1 = axes[0, 0]
            bars1 = ax1.bar(methods, silhouette_scores, color=[self.colors.get(m, '#1f77b4') for m in methods])
            ax1.set_title('Silhouette Score Comparison', fontweight='bold')
            ax1.set_ylabel('Silhouette Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars1, silhouette_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            # 2. Calinski-Harabasz Score Comparison
            calinski_scores = [quality_metrics[m]['calinski_harabasz_score'] for m in methods]
            
            ax2 = axes[0, 1]
            bars2 = ax2.bar(methods, calinski_scores, color=[self.colors.get(m, '#ff7f0e') for m in methods])
            ax2.set_title('Calinski-Harabasz Score Comparison', fontweight='bold')
            ax2.set_ylabel('Calinski-Harabasz Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars2, calinski_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{score:.1f}', ha='center', va='bottom')
            
            # 3. Davies-Bouldin Score Comparison (lower is better)
            davies_scores = [quality_metrics[m]['davies_bouldin_score'] for m in methods]
            
            ax3 = axes[1, 0]
            bars3 = ax3.bar(methods, davies_scores, color=[self.colors.get(m, '#2ca02c') for m in methods])
            ax3.set_title('Davies-Bouldin Score Comparison (Lower is Better)', fontweight='bold')
            ax3.set_ylabel('Davies-Bouldin Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars3, davies_scores):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            # 4. Overall Quality Score
            overall_scores = [quality_metrics[m]['overall_score'] for m in methods]
            
            ax4 = axes[1, 1]
            bars4 = ax4.bar(methods, overall_scores, color=[self.colors.get(m, '#d62728') for m in methods])
            ax4.set_title('Overall Quality Score', fontweight='bold')
            ax4.set_ylabel('Overall Score')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars4, overall_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"embedding_quality_analysis.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved embedding quality analysis to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating embedding quality analysis: {e}")
            return ""
    
    def _calculate_embedding_quality_metrics(self, embedding_matrix: np.ndarray, 
                                           labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics for embeddings.
        
        Args:
            embedding_matrix: Matrix of embeddings
            labels: Binary labels
        
        Returns:
            Dictionary of quality metrics
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Ensure labels are 1D
            if labels.ndim > 1:
                if labels.shape[1] == 2:  # One-hot encoded
                    labels = labels[:, 1]  # Take the second column (anomalous class)
                else:
                    labels = labels.flatten()
            
            # Ensure we have at least 2 classes
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logger.warning(f"Only {len(unique_labels)} unique labels found, cannot calculate clustering metrics")
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': 1.0,
                    'overall_score': 0.0
                }
            
            # Standardize embeddings
            scaler = StandardScaler()
            embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)
            
            # Calculate metrics
            silhouette = silhouette_score(embedding_matrix_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(embedding_matrix_scaled, labels)
            davies_bouldin = davies_bouldin_score(embedding_matrix_scaled, labels)
            
            # Calculate overall score (normalized combination)
            # Normalize Davies-Bouldin (lower is better, so invert)
            davies_bouldin_norm = 1 / (1 + davies_bouldin)
            
            # Overall score as weighted average
            overall_score = (silhouette + calinski_harabasz / 1000 + davies_bouldin_norm) / 3
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 1.0,
                'overall_score': 0.0
            } 