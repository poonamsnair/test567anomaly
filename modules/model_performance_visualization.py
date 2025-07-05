"""
Model Performance Visualization Module for AI Agent Trajectory Anomaly Detection.

Handles:
- Model performance comparison (bar charts)
- ROC/PR curves
- Threshold calibration
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve
)

from .utils import ensure_directory

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


class ModelPerformanceVisualizer:
    def __init__(self, config: Dict, output_dir: str = "charts"):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        ensure_directory(str(self.output_dir))
        
        # Setup matplotlib
        setup_matplotlib_for_plotting()
        
        # Figure settings
        self.figure_settings = self.viz_config.get('figure_settings', {})
        self.dpi = self.figure_settings.get('dpi', 300)
        self.format = self.figure_settings.get('format', 'png')
        self.bbox_inches = self.figure_settings.get('bbox_inches', 'tight')
        
        logger.info("ModelPerformanceVisualizer initialized, output dir: %s", self.output_dir)

    def plot_model_performance_comparison(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Plot model performance comparison.
        
        Args:
            evaluation_results: Dictionary of evaluation results for each model
        
        Returns:
            Path to saved figure
        """
        logger.info("Plotting model performance comparison")
        try:
            # Extract metrics for each model
            metrics = ['precision', 'recall', 'f1', 'accuracy', 'auc_roc', 'auc_pr']
            data = []
            
            # Handle different data structures
            if 'model_comparison' in evaluation_results:
                # Structure from evaluation report
                model_data = evaluation_results['model_comparison']
            elif 'model_results' in evaluation_results:
                # Structure from evaluation results
                model_data = evaluation_results['model_results']
            else:
                # Direct structure
                model_data = evaluation_results
            
            for model_name, model_metrics in model_data.items():
                if isinstance(model_metrics, dict):
                    row = {'Model': model_name}
                    
                    # Handle different metric structures
                    if 'test_metrics' in model_metrics:
                        # Nested structure
                        metrics_dict = model_metrics['test_metrics']
                    else:
                        # Direct structure
                        metrics_dict = model_metrics
                    
                    for metric in metrics:
                        value = metrics_dict.get(metric, None)
                        # Handle infinity values and convert string metrics to float
                        if value == float('inf') or value == 'inf':
                            value = 0.0
                        elif isinstance(value, str):
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                value = 0.0
                        elif value is None:
                            value = 0.0
                        row[metric] = value
                    data.append(row)
            
            if not data:
                logger.warning("No model metrics to plot")
                return ""
            
            df = pd.DataFrame(data)
            df.set_index('Model', inplace=True)
            
            # Log the data being plotted for debugging
            logger.info("Model performance data for plotting:")
            for model_name, row in df.iterrows():
                logger.info(f"{model_name}: {dict(row)}")
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            df.plot(kind='bar', ax=plt.gca(), alpha=0.7)
            plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Models', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filepath = self.output_dir / f"model_performance_comparison.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved model performance comparison to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.error(f"Error in plot_model_performance_comparison: {e}")
            return ""

    def plot_roc_curves(self, model_results: Dict[str, Any], 
                       test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot ROC curves for all models including ensemble.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved ROC curves plot
        """
        try:
            logger.info("Plotting ROC curves")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(12, 8))
            
            # Handle duplicate columns
            if test_df.columns.duplicated().any():
                logger.warning("Duplicate columns found in test_df, removing duplicates")
                test_df = test_df.loc[:, ~test_df.columns.duplicated()]
            
            # Get true labels
            if 'is_anomalous' not in test_df.columns:
                logger.error("Column 'is_anomalous' not found in test_df")
                return ""
            
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Validate inputs
            if test_features is None or len(test_features) == 0:
                logger.error("test_features is None or empty")
                return ""
            
            if len(y_true) != len(test_features):
                logger.error(f"Length mismatch: y_true={len(y_true)}, test_features={len(test_features)}")
                return ""
            
            logger.info(f"Plotting ROC curves for {len(model_results)} models")
            logger.debug(f"test_features shape: {test_features.shape}")
            logger.debug(f"y_true shape: {y_true.shape}")
            
            # Track successful plots
            successful_plots = 0
            
            # Plot ROC curves for each model
            for model_name, model_data in model_results.items():
                try:
                    model_name_str = str(model_name)
                    logger.debug(f"Processing model: {model_name_str}")
                    
                    # Check if model has required components
                    has_model = (
                        model_data.get('model') is not None or 
                        model_data.get('ensemble_model') is not None
                    )
                    
                    if not has_model:
                        logger.warning(f"Skipping {model_name_str}: No model object present")
                        continue
                    
                    # Get predictions from the model
                    scores = self._get_model_scores(model_data, test_features, model_name_str)
                    
                    if scores is None or len(scores) == 0:
                        logger.warning(f"No valid scores for {model_name_str}")
                        continue
                    
                    if len(scores) != len(y_true):
                        logger.warning(f"Score length mismatch for {model_name_str}: {len(scores)} vs {len(y_true)}")
                        continue
                    
                    # Calculate ROC curve
                    try:
                        fpr, tpr, _ = roc_curve(y_true, scores)
                        auc_score = auc(fpr, tpr)
                        
                        # Plot the curve with distinct colors
                        plt.plot(fpr, tpr, label=f'{model_name_str} (AUC = {auc_score:.3f})', 
                               linewidth=2.5, alpha=0.8)
                        
                        successful_plots += 1
                        logger.info(f"Successfully plotted ROC curve for {model_name_str} (AUC: {auc_score:.3f})")
                        
                    except Exception as e:
                        logger.error(f"Error calculating ROC for {model_name_str}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing model {model_name}: {e}")
                    continue
            
            if successful_plots == 0:
                logger.error("No successful ROC curves plotted")
                plt.close()
                return ""
            
            # Add diagonal line (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5, linewidth=1.5)
            
            # Customize plot
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curves for Anomaly Detection Models', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10, loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            # Save plot
            filepath = self.output_dir / f"roc_curves.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved ROC curves to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_roc_curves: {e}")
            plt.close()
            return ""

    def _get_model_scores(self, model_data: Dict[str, Any], test_features: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """
        Helper method to get anomaly scores from a model.
        
        Args:
            model_data: Model data dictionary
            test_features: Test features array
            model_name: Name of the model for logging
            
        Returns:
            Anomaly scores array or None if failed
        """
        try:
            # Handle ensemble model
            if model_data.get('ensemble_model') is not None:
                ensemble_model = model_data['ensemble_model']
                scores, _ = ensemble_model.predict_anomalies(test_features)
                return scores
            
            # Handle standard models
            model = model_data.get('model')
            method = model_data.get('method')
            
            if model is None:
                logger.error(f"No model found in model_data for {model_name}")
                return None
            
            # Get predictions based on model type
            if method == 'isolation_forest':
                scores = model.score_samples(test_features)
            elif method == 'one_class_svm':
                # Apply same preprocessing as during training
                scaler = model_data.get('scaler')
                pca = model_data.get('pca')
                variance_threshold = model_data.get('variance_threshold')
                
                if scaler is None:
                    logger.error(f"No scaler found for {model_name}")
                    return None
                
                # Apply preprocessing
                test_features_processed = test_features.copy()
                
                # Apply low variance feature removal if it was used during training
                if variance_threshold is not None:
                    test_features_processed = variance_threshold.transform(test_features_processed)
                
                # Apply PCA transformation if it was used during training
                if pca is not None:
                    test_features_processed = pca.transform(test_features_processed)
                
                # Scale features
                test_features_scaled = scaler.transform(test_features_processed)
                
                # Get decision scores
                scores = model.decision_function(test_features_scaled)
            elif method == 'gnn_autoencoder':
                # GNN models require graph data, not flat features
                # We need to get the test graphs from the model results
                test_graphs = model_data.get('test_graphs')
                anomaly_detector = model_data.get('anomaly_detector')
                
                if test_graphs is None:
                    logger.warning(f"No test graphs available for GNN autoencoder {model_name}. Returning None.")
                    return None
                
                if anomaly_detector is None:
                    logger.warning(f"No anomaly detector available for GNN autoencoder {model_name}. Returning None.")
                    return None
                
                try:
                    # Use the model's prediction method for graphs
                    scores, _ = anomaly_detector.predict_anomalies_graphs(
                        model_data, test_graphs
                    )
                    return scores
                except Exception as e:
                    logger.error(f"Error getting GNN scores for {model_name}: {e}")
                    return None
            else:
                logger.error(f"Unknown model method: {method} for {model_name}")
                return None
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting scores for {model_name}: {e}")
            return None

    def plot_precision_recall_curves(self, model_results: Dict[str, Any], 
                                   test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved Precision-Recall curves plot
        """
        try:
            logger.info("Plotting Precision-Recall curves")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(12, 8))
            
            # Handle duplicate columns
            if test_df.columns.duplicated().any():
                logger.warning("Duplicate columns found in test_df, removing duplicates")
                test_df = test_df.loc[:, ~test_df.columns.duplicated()]
            
            # Get true labels
            if 'is_anomalous' not in test_df.columns:
                logger.error("Column 'is_anomalous' not found in test_df")
                return ""
            
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Validate inputs
            if test_features is None or len(test_features) == 0:
                logger.error("test_features is None or empty")
                return ""
            
            if len(y_true) != len(test_features):
                logger.error(f"Length mismatch: y_true={len(y_true)}, test_features={len(test_features)}")
                return ""
            
            logger.info(f"Plotting Precision-Recall curves for {len(model_results)} models")
            
            # Track successful plots
            successful_plots = 0
            
            # Plot Precision-Recall curves for each model
            for model_name, model_data in model_results.items():
                try:
                    model_name_str = str(model_name)
                    logger.debug(f"Processing model: {model_name_str}")
                    
                    # Check if model has required components
                    has_model = (
                        model_data.get('model') is not None or 
                        model_data.get('ensemble_model') is not None
                    )
                    
                    if not has_model:
                        logger.warning(f"Skipping {model_name_str}: No model object present")
                        continue
                    
                    # Get predictions from the model
                    scores = self._get_model_scores(model_data, test_features, model_name_str)
                    
                    if scores is None or len(scores) == 0:
                        logger.warning(f"No valid scores for {model_name_str}")
                        continue
                    
                    if len(scores) != len(y_true):
                        logger.warning(f"Score length mismatch for {model_name_str}: {len(scores)} vs {len(y_true)}")
                        continue
                    
                    # Calculate Precision-Recall curve
                    try:
                        precision, recall, _ = precision_recall_curve(y_true, scores)
                        auc_score = auc(recall, precision)
                        
                        # Plot the curve with distinct colors
                        plt.plot(recall, precision, label=f'{model_name_str} (AUC = {auc_score:.3f})', 
                               linewidth=2.5, alpha=0.8)
                        
                        successful_plots += 1
                        logger.info(f"Successfully plotted Precision-Recall curve for {model_name_str} (AUC: {auc_score:.3f})")
                        
                    except Exception as e:
                        logger.error(f"Error calculating Precision-Recall for {model_name_str}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing model {model_name}: {e}")
                    continue
            
            if successful_plots == 0:
                logger.error("No successful Precision-Recall curves plotted")
                plt.close()
                return ""
            
            # Add baseline (random classifier)
            baseline = len(y_true[y_true == 1]) / len(y_true)
            plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random Classifier (P = {baseline:.3f})', alpha=0.5)
            
            # Customize plot
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves for Anomaly Detection Models', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10, loc='lower left')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            # Save plot
            filepath = self.output_dir / f"precision_recall_curves.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved precision-recall curves to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_precision_recall_curves: {e}")
            plt.close()
            return ""

    def plot_threshold_calibration(self, threshold_results: Dict[str, Dict[str, float]]) -> str:
        """
        Plot threshold calibration results.
        
        Args:
            threshold_results: Dictionary of threshold calibration results
        
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting threshold calibration results")
            
            if not threshold_results:
                logger.warning("No threshold results to plot")
                return ""
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Threshold values comparison
            ax1 = axes[0, 0]
            models = list(threshold_results.keys())
            thresholds = [threshold_results[model].get('threshold', 0) for model in models]
            
            bars = ax1.bar(models, thresholds, alpha=0.7, color='skyblue')
            ax1.set_title('Optimal Threshold Values')
            ax1.set_ylabel('Threshold Value')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, threshold in zip(bars, thresholds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(thresholds) * 0.01,
                        f'{threshold:.3f}', ha='center', va='bottom')
            
            # 2. F1 scores at optimal thresholds
            ax2 = axes[0, 1]
            f1_scores = [threshold_results[model].get('f1_score', 0) for model in models]
            
            bars2 = ax2.bar(models, f1_scores, alpha=0.7, color='lightgreen')
            ax2.set_title('F1 Scores at Optimal Thresholds')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, f1 in zip(bars2, f1_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom')
            
            # 3. Precision vs Recall trade-off
            ax3 = axes[1, 0]
            precisions = [threshold_results[model].get('precision', 0) for model in models]
            recalls = [threshold_results[model].get('recall', 0) for model in models]
            
            scatter = ax3.scatter(precisions, recalls, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
            ax3.set_title('Precision vs Recall Trade-off')
            ax3.set_xlabel('Precision')
            ax3.set_ylabel('Recall')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model in enumerate(models):
                ax3.annotate(model, (precisions[i], recalls[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # 4. Threshold optimization summary
            ax4 = axes[1, 1]
            summary_data = []
            for model in models:
                data = threshold_results[model]
                summary_data.append({
                    'Model': model,
                    'Threshold': data.get('threshold', 0),
                    'F1': data.get('f1_score', 0),
                    'Precision': data.get('precision', 0),
                    'Recall': data.get('recall', 0)
                })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.set_index('Model', inplace=True)
                
                # Create heatmap
                im = ax4.imshow(df.values, cmap='YlOrRd', aspect='auto')
                ax4.set_xticks(range(len(df.columns)))
                ax4.set_yticks(range(len(df.index)))
                ax4.set_xticklabels(df.columns, rotation=45)
                ax4.set_yticklabels(df.index)
                ax4.set_title('Threshold Calibration Summary')
                
                # Add text annotations
                for i in range(len(df.index)):
                    for j in range(len(df.columns)):
                        text = ax4.text(j, i, f'{df.iloc[i, j]:.3f}',
                                       ha="center", va="center", color="black", fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.output_dir / f"threshold_calibration.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved threshold calibration results to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_threshold_calibration: {e}")
            return ""

    def plot_hyperparameter_tuning_results(self, model_results: Dict[str, Any]) -> str:
        """
        Plot hyperparameter tuning results for all models.
        
        Args:
            model_results: Dictionary containing model results with tuning history
        
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting hyperparameter tuning results")
            
            # Filter models that have tuning history
            models_with_tuning = {}
            for model_name, model_data in model_results.items():
                if 'tuning_history' in model_data or 'best_params' in model_data:
                    models_with_tuning[model_name] = model_data
            
            if not models_with_tuning:
                logger.warning("No hyperparameter tuning data found")
                return ""
            
            # Create subplots
            n_models = len(models_with_tuning)
            cols = min(2, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (model_name, model_data) in enumerate(models_with_tuning.items()):
                ax = axes[i]
                
                # Extract tuning history
                if 'tuning_history' in model_data:
                    history = model_data['tuning_history']
                    if isinstance(history, list) and len(history) > 0:
                        # Plot scores over iterations
                        scores = [h.get('score', 0) for h in history]
                        iterations = range(1, len(scores) + 1)
                        
                        ax.plot(iterations, scores, 'o-', alpha=0.7, linewidth=2)
                        ax.set_title(f'{model_name} Hyperparameter Tuning')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Score')
                        ax.grid(True, alpha=0.3)
                        
                        # Add best score annotation - handle different score types
                        if model_name == 'isolation_forest':
                            # For isolation forest, lower scores are better
                            best_score = min(scores)
                            best_iter = scores.index(best_score) + 1
                        else:
                            # For other models, higher scores are better
                            best_score = max(scores)
                            best_iter = scores.index(best_score) + 1
                        
                        ax.annotate(f'Best: {best_score:.3f}', 
                                  xy=(best_iter, best_score),
                                  xytext=(best_iter + 1, best_score + 0.1),
                                  arrowprops=dict(arrowstyle='->', color='red'),
                                  fontsize=10, color='red')
                    else:
                        ax.text(0.5, 0.5, 'No tuning history', ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(f'{model_name} Hyperparameter Tuning')
                else:
                    ax.text(0.5, 0.5, 'No tuning data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{model_name} Hyperparameter Tuning')
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"hyperparameter_tuning_results.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved hyperparameter tuning results to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating hyperparameter tuning results: {e}")
            return ""

    def plot_hyperparameter_sensitivity_analysis(self, model_results: Dict[str, Any]) -> str:
        """
        Plot hyperparameter sensitivity analysis.
        
        Args:
            model_results: Dictionary containing model results with parameter sensitivity data
        
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting hyperparameter sensitivity analysis")
            
            # Filter models that have tuning history or best params
            models_with_data = {}
            for model_name, model_data in model_results.items():
                if 'tuning_history' in model_data or 'best_params' in model_data:
                    models_with_data[model_name] = model_data
            
            if not models_with_data:
                logger.warning("No hyperparameter data found")
                return ""
            
            # Create subplots
            n_models = len(models_with_data)
            cols = min(2, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (model_name, model_data) in enumerate(models_with_data.items()):
                ax = axes[i]
                
                # Calculate parameter sensitivity from tuning history
                if 'tuning_history' in model_data and model_data['tuning_history']:
                    history = model_data['tuning_history']
                    
                    # Extract all unique parameters
                    all_params = set()
                    for entry in history:
                        all_params.update(entry['params'].keys())
                    
                    # Calculate sensitivity for each parameter
                    param_sensitivity = {}
                    for param in all_params:
                        # Group scores by parameter value
                        param_groups = {}
                        for entry in history:
                            if param in entry['params']:
                                param_value = str(entry['params'][param])
                                if param_value not in param_groups:
                                    param_groups[param_value] = []
                                param_groups[param_value].append(entry['score'])
                        
                        if len(param_groups) > 1:  # Only if parameter varies
                            # Calculate mean score for each parameter value
                            param_means = {val: np.mean(scores) for val, scores in param_groups.items()}
                            
                            # Calculate sensitivity as the range of mean scores
                            mean_scores = list(param_means.values())
                            if len(mean_scores) > 1:
                                score_range = max(mean_scores) - min(mean_scores)
                                overall_mean = np.mean(mean_scores)
                                if overall_mean != 0:
                                    sensitivity = score_range / abs(overall_mean)
                                    param_sensitivity[param] = sensitivity
                    
                    if param_sensitivity:
                        # Sort by sensitivity
                        sorted_params = sorted(param_sensitivity.items(), key=lambda x: x[1], reverse=True)
                        params = [p[0] for p in sorted_params]
                        importance = [p[1] for p in sorted_params]
                        
                        # Create bar plot of parameter importance
                        bars = ax.barh(params, importance, alpha=0.7)
                        ax.set_title(f'{model_name} Parameter Sensitivity')
                        ax.set_xlabel('Sensitivity Score (CV)')
                        
                        # Add value labels
                        for bar, imp in zip(bars, importance):
                            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{imp:.3f}', ha='left', va='center')
                    else:
                        ax.text(0.5, 0.5, 'No varying parameters', ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(f'{model_name} Parameter Sensitivity')
                else:
                    ax.text(0.5, 0.5, 'No tuning history', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{model_name} Parameter Sensitivity')
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"hyperparameter_sensitivity_analysis.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved hyperparameter sensitivity analysis to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating hyperparameter sensitivity analysis: {e}")
            return ""

    def plot_training_loss_curves(self, model_results: Dict[str, Any]) -> str:
        """
        Plot training loss curves for models that have training history.
        
        Args:
            model_results: Dictionary containing model results with training history
        
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting training loss curves")
            
            # Filter models that have training history
            models_with_history = {}
            for model_name, model_data in model_results.items():
                if 'training_history' in model_data or 'loss_history' in model_data:
                    models_with_history[model_name] = model_data
            
            if not models_with_history:
                logger.warning("No training history data found")
                return ""
            
            # Create subplots
            n_models = len(models_with_history)
            cols = min(2, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (model_name, model_data) in enumerate(models_with_history.items()):
                ax = axes[i]
                
                # Extract training history
                history = model_data.get('training_history', model_data.get('loss_history', {}))
                
                if isinstance(history, dict) and len(history) > 0:
                    # Plot training and validation loss if available
                    epochs = range(1, len(history.get('train_loss', [])) + 1)
                    
                    if 'train_loss' in history:
                        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
                    
                    if 'val_loss' in history:
                        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                    
                    ax.set_title(f'{model_name} Training Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add final loss annotation
                    if 'train_loss' in history and len(history['train_loss']) > 0:
                        final_loss = history['train_loss'][-1]
                        ax.annotate(f'Final: {final_loss:.4f}', 
                                  xy=(len(epochs), final_loss),
                                  xytext=(len(epochs) - 2, final_loss + 0.1),
                                  arrowprops=dict(arrowstyle='->', color='blue'),
                                  fontsize=10, color='blue')
                else:
                    ax.text(0.5, 0.5, 'No training history', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{model_name} Training Loss')
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"training_loss_curves.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved training loss curves to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating training loss curves: {e}")
            return ""

    def plot_before_vs_after_analysis(self, model_results: Dict[str, Any]) -> str:
        """
        Plot before vs after analysis comparing default vs tuned models.
        
        Args:
            model_results: Dictionary containing model results with before/after data
        
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting before vs after analysis")
            
            # Extract before and after metrics
            models_with_comparison = {}
            for model_name, model_data in model_results.items():
                if ('default_metrics' in model_data or 'before_tuning' in model_data) and \
                   ('test_metrics' in model_data or 'after_tuning' in model_data):
                    models_with_comparison[model_name] = model_data
            
            if not models_with_comparison:
                logger.warning("No before/after comparison data found")
                return ""
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            metrics = ['f1', 'precision', 'recall', 'accuracy']
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                model_names = []
                before_scores = []
                after_scores = []
                
                for model_name, model_data in models_with_comparison.items():
                    # Get before scores
                    before_data = model_data.get('default_metrics', model_data.get('before_tuning', {}))
                    before_score = before_data.get(metric, 0.0)
                    
                    # Get after scores
                    after_data = model_data.get('test_metrics', model_data.get('after_tuning', {}))
                    after_score = after_data.get(metric, 0.0)
                    
                    model_names.append(model_name)
                    before_scores.append(before_score)
                    after_scores.append(after_score)
                
                if model_names:
                    x = np.arange(len(model_names))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, before_scores, width, label='Before Tuning', alpha=0.7)
                    bars2 = ax.bar(x + width/2, after_scores, width, label='After Tuning', alpha=0.7)
                    
                    ax.set_title(f'{metric.upper()} Score Comparison')
                    ax.set_ylabel('Score')
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=45)
                    ax.legend()
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{metric.upper()} Score Comparison')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"before_vs_after_analysis.{self.format}"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info("Saved before vs after analysis to %s", filepath)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating before vs after analysis: {e}")
            return ""