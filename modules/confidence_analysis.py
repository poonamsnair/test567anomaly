"""
Model Confidence Analysis Module for AI Agent Trajectory Anomaly Detection.

Handles:
- Confidence distribution plots
- Confidence vs performance correlation analysis
- Model calibration plots
- Confidence-based error analysis
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy import stats

from .utils import ensure_directory

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    """
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


class ConfidenceAnalyzer:
    """
    Comprehensive confidence analysis for anomaly detection models.
    
    This class provides methods to analyze model confidence including:
    1. Confidence distribution analysis
    2. Confidence vs performance correlation
    3. Model calibration analysis
    4. Confidence-based error analysis
    """
    
    def __init__(self, config: Dict, output_dir: str = "charts"):
        """
        Initialize confidence analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save charts
        """
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
        
        # Color scheme for models
        self.model_colors = {
            'isolation_forest': '#1f77b4',
            'one_class_svm': '#ff7f0e',
            'local_outlier_factor': '#2ca02c',
            'elliptic_envelope': '#d62728',
            'ensemble_model': '#9467bd',
            'autoencoder': '#8c564b',
            'gaussian_mixture': '#e377c2'
        }
        
        logger.info("ConfidenceAnalyzer initialized, output dir: %s", self.output_dir)

    def plot_confidence_distributions(self, model_results: Dict[str, Any], 
                                    test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot confidence score distributions for all models.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting confidence distributions")
            
            # Get true labels
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Create subplots
            n_models = len([k for k, v in model_results.items() if 'error' not in v])
            if n_models == 0:
                logger.warning("No valid models found for confidence analysis")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            plot_idx = 0
            successful_plots = 0
            
            for model_name, model_data in model_results.items():
                if 'error' in model_data:
                    continue
                
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                try:
                    # Get confidence scores
                    confidence_scores = self._get_confidence_scores(model_data, test_features, model_name)
                    
                    if confidence_scores is None:
                        continue
                    
                    # Separate scores by true labels
                    normal_scores = confidence_scores[y_true == 0]
                    anomalous_scores = confidence_scores[y_true == 1]
                    
                    # Plot distributions
                    if len(normal_scores) > 0:
                        ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', 
                               color='green', density=True)
                    if len(anomalous_scores) > 0:
                        ax.hist(anomalous_scores, bins=30, alpha=0.6, label='Anomalous', 
                               color='red', density=True)
                    
                    # Add kernel density estimation
                    if len(normal_scores) > 0:
                        sns.kdeplot(normal_scores, ax=ax, color='darkgreen', linewidth=2)
                    if len(anomalous_scores) > 0:
                        sns.kdeplot(anomalous_scores, ax=ax, color='darkred', linewidth=2)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Confidence Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    stats_text = f'Normal: μ={np.mean(normal_scores):.3f}, σ={np.std(normal_scores):.3f}\n'
                    stats_text += f'Anomalous: μ={np.mean(anomalous_scores):.3f}, σ={np.std(anomalous_scores):.3f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    successful_plots += 1
                    plot_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error plotting confidence distribution for {model_name}: {e}")
                    continue
            
            # Hide unused subplots
            for i in range(successful_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.output_dir / f"confidence_distributions.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved confidence distributions to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_confidence_distributions: {e}")
            plt.close()
            return ""

    def plot_confidence_vs_performance(self, model_results: Dict[str, Any], 
                                     test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot confidence vs performance correlation analysis.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting confidence vs performance analysis")
            
            # Get true labels
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Create subplots
            n_models = len([k for k, v in model_results.items() if 'error' not in v])
            if n_models == 0:
                logger.warning("No valid models found for confidence analysis")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            plot_idx = 0
            successful_plots = 0
            
            for model_name, model_data in model_results.items():
                if 'error' in model_data:
                    continue
                
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                try:
                    # Get confidence scores
                    confidence_scores = self._get_confidence_scores(model_data, test_features, model_name)
                    
                    if confidence_scores is None:
                        continue
                    
                    # Calculate performance metrics for different confidence thresholds
                    thresholds = np.linspace(np.min(confidence_scores), np.max(confidence_scores), 50)
                    precisions = []
                    recalls = []
                    f1_scores = []
                    
                    for threshold in thresholds:
                        # For anomaly detection, higher scores typically indicate anomalies
                        # We'll use the threshold to determine predictions
                        y_pred = (confidence_scores >= threshold).astype(int)
                        
                        # Calculate metrics
                        tp = np.sum((y_true == 1) & (y_pred == 1))
                        fp = np.sum((y_true == 0) & (y_pred == 1))
                        fn = np.sum((y_true == 1) & (y_pred == 0))
                        tn = np.sum((y_true == 0) & (y_pred == 0))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)
                    
                    # Plot confidence vs performance
                    ax.plot(thresholds, precisions, label='Precision', linewidth=2, alpha=0.8)
                    ax.plot(thresholds, recalls, label='Recall', linewidth=2, alpha=0.8)
                    ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2, alpha=0.8)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Confidence vs Performance')
                    ax.set_xlabel('Confidence Threshold')
                    ax.set_ylabel('Performance Metric')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add optimal threshold annotation with better positioning
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    optimal_f1 = f1_scores[optimal_idx]
                    
                    # Add vertical line for optimal threshold
                    ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                    
                    # Calculate annotation position to stay within plot bounds
                    x_range = np.max(thresholds) - np.min(thresholds)
                    y_range = max(f1_scores) - min(f1_scores)
                    
                    # Position annotation to the left or right of the line based on available space
                    if optimal_threshold > (np.min(thresholds) + x_range * 0.7):
                        # Position to the left if threshold is in the right portion
                        text_x = optimal_threshold - x_range * 0.15
                        arrow_x = optimal_threshold - x_range * 0.02
                    else:
                        # Position to the right if threshold is in the left portion
                        text_x = optimal_threshold + x_range * 0.15
                        arrow_x = optimal_threshold + x_range * 0.02
                    
                    # Ensure text stays within plot bounds
                    text_x = np.clip(text_x, np.min(thresholds), np.max(thresholds))
                    text_y = np.clip(optimal_f1 + y_range * 0.1, min(f1_scores), max(f1_scores))
                    
                    ax.annotate(f'Optimal F1: {optimal_f1:.3f}\nThreshold: {optimal_threshold:.3f}',
                              xy=(optimal_threshold, optimal_f1),
                              xytext=(text_x, text_y),
                              arrowprops=dict(arrowstyle='->', color='red', lw=1),
                              fontsize=8, color='red',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    # Set reasonable x-axis limits to prevent excessive width
                    x_margin = x_range * 0.05
                    ax.set_xlim(np.min(thresholds) - x_margin, np.max(thresholds) + x_margin)
                    
                    successful_plots += 1
                    plot_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error plotting confidence vs performance for {model_name}: {e}")
                    continue
            
            # Hide unused subplots
            for i in range(successful_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.output_dir / f"confidence_vs_performance.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved confidence vs performance to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_confidence_vs_performance: {e}")
            plt.close()
            return ""

    def plot_calibration_analysis(self, model_results: Dict[str, Any], 
                                test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot model calibration analysis.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting calibration analysis")
            
            # Get true labels
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Create subplots
            n_models = len([k for k, v in model_results.items() if 'error' not in v])
            if n_models == 0:
                logger.warning("No valid models found for calibration analysis")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            plot_idx = 0
            successful_plots = 0
            
            for model_name, model_data in model_results.items():
                if 'error' in model_data:
                    continue
                
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                try:
                    # Get confidence scores
                    confidence_scores = self._get_confidence_scores(model_data, test_features, model_name)
                    
                    if confidence_scores is None:
                        continue
                    
                    # Convert scores to probabilities (0-1 range)
                    # For anomaly detection, we need to normalize scores
                    scores_normalized = self._normalize_scores_to_probabilities(confidence_scores)
                    
                    # Calculate calibration curve
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true, scores_normalized, n_bins=10
                    )
                    
                    # Plot calibration curve
                    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                           label=f'{model_name.replace("_", " ").title()}', 
                           linewidth=2, markersize=8)
                    
                    # Plot perfect calibration line
                    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Calibration')
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    
                    # Calculate Brier score
                    brier_score = brier_score_loss(y_true, scores_normalized)
                    
                    # Add Brier score annotation
                    ax.text(0.02, 0.98, f'Brier Score: {brier_score:.4f}', 
                           transform=ax.transAxes, verticalalignment='top', 
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                    successful_plots += 1
                    plot_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error plotting calibration for {model_name}: {e}")
                    continue
            
            # Hide unused subplots
            for i in range(successful_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.output_dir / f"calibration_analysis.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved calibration analysis to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_calibration_analysis: {e}")
            plt.close()
            return ""

    def plot_confidence_error_analysis(self, model_results: Dict[str, Any], 
                                     test_df: pd.DataFrame, test_features: np.ndarray) -> str:
        """
        Plot confidence-based error analysis.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved figure
        """
        try:
            logger.info("Plotting confidence error analysis")
            
            # Get true labels
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Create subplots
            n_models = len([k for k, v in model_results.items() if 'error' not in v])
            if n_models == 0:
                logger.warning("No valid models found for error analysis")
                return ""
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            plot_idx = 0
            successful_plots = 0
            
            for model_name, model_data in model_results.items():
                if 'error' in model_data:
                    continue
                
                if plot_idx >= len(axes):
                    break
                
                ax = axes[plot_idx]
                
                try:
                    # Get confidence scores
                    confidence_scores = self._get_confidence_scores(model_data, test_features, model_name)
                    
                    if confidence_scores is None:
                        continue
                    
                    # Get predictions using default threshold (median)
                    threshold = np.median(confidence_scores)
                    y_pred = (confidence_scores >= threshold).astype(int)
                    
                    # Identify errors
                    false_positives = (y_true == 0) & (y_pred == 1)
                    false_negatives = (y_true == 1) & (y_pred == 0)
                    true_positives = (y_true == 1) & (y_pred == 1)
                    true_negatives = (y_true == 0) & (y_pred == 0)
                    
                    # Plot confidence distributions for different prediction types
                    if np.any(true_negatives):
                        ax.hist(confidence_scores[true_negatives], bins=20, alpha=0.6, 
                               label='True Negatives', color='green', density=True)
                    if np.any(true_positives):
                        ax.hist(confidence_scores[true_positives], bins=20, alpha=0.6, 
                               label='True Positives', color='blue', density=True)
                    if np.any(false_positives):
                        ax.hist(confidence_scores[false_positives], bins=20, alpha=0.6, 
                               label='False Positives', color='orange', density=True)
                    if np.any(false_negatives):
                        ax.hist(confidence_scores[false_negatives], bins=20, alpha=0.6, 
                               label='False Negatives', color='red', density=True)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Confidence Error Analysis')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add error statistics
                    error_stats = f'FP: {np.sum(false_positives)}\n'
                    error_stats += f'FN: {np.sum(false_negatives)}\n'
                    error_stats += f'TP: {np.sum(true_positives)}\n'
                    error_stats += f'TN: {np.sum(true_negatives)}'
                    
                    ax.text(0.02, 0.98, error_stats, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    successful_plots += 1
                    plot_idx += 1
                    
                except Exception as e:
                    logger.error(f"Error plotting error analysis for {model_name}: {e}")
                    continue
            
            # Hide unused subplots
            for i in range(successful_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            filepath = self.output_dir / f"confidence_error_analysis.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved confidence error analysis to {filepath} ({successful_plots} models)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in plot_confidence_error_analysis: {e}")
            plt.close()
            return ""

    def _get_confidence_scores(self, model_data: Dict[str, Any], 
                             test_features: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """
        Helper method to get confidence scores from a model.
        
        Args:
            model_data: Model data dictionary
            test_features: Test features array
            model_name: Name of the model for logging
            
        Returns:
            Confidence scores array or None if failed
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
            elif method == 'local_outlier_factor':
                scores = model.score_samples(test_features)
            elif method == 'elliptic_envelope':
                scores = model.score_samples(test_features)
            elif method == 'gaussian_mixture':
                scores = model.score_samples(test_features)
            else:
                logger.warning(f"Unknown method {method} for {model_name}")
                return None
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting confidence scores for {model_name}: {e}")
            return None

    def _normalize_scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to probability range [0, 1].
        
        Args:
            scores: Raw anomaly scores
            
        Returns:
            Normalized probabilities
        """
        try:
            # Use sigmoid function to normalize scores to [0, 1]
            # First, standardize the scores
            scores_std = (scores - np.mean(scores)) / np.std(scores)
            
            # Apply sigmoid function
            probabilities = 1 / (1 + np.exp(-scores_std))
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            # Fallback to min-max normalization
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def generate_comprehensive_confidence_report(self, model_results: Dict[str, Any], 
                                               test_df: pd.DataFrame, 
                                               test_features: np.ndarray) -> str:
        """
        Generate a comprehensive confidence analysis report.
        
        Args:
            model_results: Dictionary containing model results
            test_df: Test DataFrame with labels
            test_features: Test features array
            
        Returns:
            Path to saved comprehensive report
        """
        try:
            logger.info("Generating comprehensive confidence report")
            
            # Create a large figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 1. Confidence Distributions (top row)
            ax1 = fig.add_subplot(gs[0, :2])
            ax2 = fig.add_subplot(gs[0, 2:])
            
            # 2. Confidence vs Performance (second row)
            ax3 = fig.add_subplot(gs[1, :2])
            ax4 = fig.add_subplot(gs[1, 2:])
            
            # 3. Calibration Analysis (third row)
            ax5 = fig.add_subplot(gs[2, :2])
            ax6 = fig.add_subplot(gs[2, 2:])
            
            # 4. Error Analysis (bottom row)
            ax7 = fig.add_subplot(gs[3, :2])
            ax8 = fig.add_subplot(gs[3, 2:])
            
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
            
            # Get true labels
            y_true = test_df['is_anomalous'].astype(int).values
            
            # Plot for first two models
            model_count = 0
            for model_name, model_data in model_results.items():
                if 'error' in model_data or model_count >= 2:
                    continue
                
                try:
                    # Get confidence scores
                    confidence_scores = self._get_confidence_scores(model_data, test_features, model_name)
                    
                    if confidence_scores is None:
                        continue
                    
                    # Plot 1: Confidence Distribution
                    ax = axes[model_count]
                    normal_scores = confidence_scores[y_true == 0]
                    anomalous_scores = confidence_scores[y_true == 1]
                    
                    if len(normal_scores) > 0:
                        ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', 
                               color='green', density=True)
                    if len(anomalous_scores) > 0:
                        ax.hist(anomalous_scores, bins=30, alpha=0.6, label='Anomalous', 
                               color='red', density=True)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Confidence Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 2: Confidence vs Performance
                    ax = axes[model_count + 2]
                    thresholds = np.linspace(np.min(confidence_scores), np.max(confidence_scores), 50)
                    f1_scores = []
                    
                    for threshold in thresholds:
                        y_pred = (confidence_scores >= threshold).astype(int)
                        tp = np.sum((y_true == 1) & (y_pred == 1))
                        fp = np.sum((y_true == 0) & (y_pred == 1))
                        fn = np.sum((y_true == 1) & (y_pred == 0))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        f1_scores.append(f1)
                    
                    ax.plot(thresholds, f1_scores, linewidth=2, alpha=0.8)
                    ax.set_title(f'{model_name.replace("_", " ").title()} F1 vs Threshold')
                    ax.set_xlabel('Confidence Threshold')
                    ax.set_ylabel('F1-Score')
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 3: Calibration
                    ax = axes[model_count + 4]
                    scores_normalized = self._normalize_scores_to_probabilities(confidence_scores)
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true, scores_normalized, n_bins=10
                    )
                    
                    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                           linewidth=2, markersize=8)
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_title(f'{model_name.replace("_", " ").title()} Calibration')
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    
                    # Plot 4: Error Analysis
                    ax = axes[model_count + 6]
                    threshold = np.median(confidence_scores)
                    y_pred = (confidence_scores >= threshold).astype(int)
                    
                    false_positives = (y_true == 0) & (y_pred == 1)
                    false_negatives = (y_true == 1) & (y_pred == 0)
                    true_positives = (y_true == 1) & (y_pred == 1)
                    true_negatives = (y_true == 0) & (y_pred == 0)
                    
                    if np.any(true_negatives):
                        ax.hist(confidence_scores[true_negatives], bins=20, alpha=0.6, 
                               label='True Negatives', color='green', density=True)
                    if np.any(true_positives):
                        ax.hist(confidence_scores[true_positives], bins=20, alpha=0.6, 
                               label='True Positives', color='blue', density=True)
                    if np.any(false_positives):
                        ax.hist(confidence_scores[false_positives], bins=20, alpha=0.6, 
                               label='False Positives', color='orange', density=True)
                    if np.any(false_negatives):
                        ax.hist(confidence_scores[false_negatives], bins=20, alpha=0.6, 
                               label='False Negatives', color='red', density=True)
                    
                    ax.set_title(f'{model_name.replace("_", " ").title()} Error Analysis')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    model_count += 1
                    
                except Exception as e:
                    logger.error(f"Error in comprehensive report for {model_name}: {e}")
                    continue
            
            # Add summary statistics
            fig.suptitle('Comprehensive Model Confidence Analysis', fontsize=16, fontweight='bold')
            
            # Save plot
            filepath = self.output_dir / f"comprehensive_confidence_analysis.{self.format}"
            plt.savefig(str(filepath), dpi=self.dpi, bbox_inches=self.bbox_inches)
            plt.close()
            
            logger.info(f"Saved comprehensive confidence analysis to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error in generate_comprehensive_confidence_report: {e}")
            plt.close()
            return "" 