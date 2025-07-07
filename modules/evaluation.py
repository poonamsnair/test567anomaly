"""
Evaluation framework for AI Agent Trajectory Anomaly Detection.

This module implements comprehensive evaluation for unsupervised anomaly detection:
- Proper data splitting (training only on normal data)
- Threshold calibration using validation set
- Comprehensive metrics calculation
- Performance analysis across different anomaly types and severities

Note: Anomaly labels are used ONLY for evaluation, never during training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

from .utils import Timer

logger = logging.getLogger(__name__)


class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluation framework for unsupervised anomaly detection.
    
    This class handles:
    1. Proper data splitting for unsupervised learning
    2. Threshold calibration using validation data
    3. Performance evaluation across multiple metrics
    4. Analysis by anomaly type and severity
    
    Key principle: Anomaly labels are used ONLY for evaluation, never training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        
        # Data split ratios
        self.train_ratio = self.eval_config.get('data_split', {}).get('train_ratio', 0.6)
        self.val_ratio = self.eval_config.get('data_split', {}).get('validation_ratio', 0.2)
        self.test_ratio = self.eval_config.get('data_split', {}).get('test_ratio', 0.2)
        
        # Threshold calibration methods
        self.calibration_methods = self.eval_config.get('threshold_calibration', [])
        
        # Metrics to calculate
        self.metrics = self.eval_config.get('metrics', [])
        
        logger.info("AnomalyDetectionEvaluator initialized")
    
    def create_unsupervised_data_splits(self, features_df: pd.DataFrame, 
                                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create proper data splits for unsupervised anomaly detection.
        
        Key principle: Training set contains only normal trajectories.
        Validation and test sets contain both normal and anomalous trajectories.
        
        Args:
            features_df: Complete feature DataFrame with anomaly labels
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating unsupervised data splits from %d trajectories", len(features_df))
        
        # Check for duplicate columns and remove them
        if features_df.columns.duplicated().any():
            logger.warning("Duplicate columns found in features_df: %s", 
                          features_df.columns[features_df.columns.duplicated()].tolist())
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]
        
        # Separate normal and anomalous trajectories
        normal_trajectories = features_df[~features_df.get('is_anomalous', False)].copy()
        anomalous_trajectories = features_df[features_df.get('is_anomalous', False)].copy()
        
        logger.info("Normal trajectories: %d, Anomalous trajectories: %d", 
                   len(normal_trajectories), len(anomalous_trajectories))

        if self.train_ratio >= 1.0 - 1e-9: # Effectively train_ratio is 1.0
            logger.warning("Train ratio is ~1.0. All normal data will be used for training. Validation and test sets will only contain anomalous data if present, or be empty.")
            train_df = normal_trajectories.copy()
            val_df = pd.DataFrame(columns=features_df.columns) # Empty df with same columns
            test_df = pd.DataFrame(columns=features_df.columns) # Empty df with same columns

            if len(anomalous_trajectories) > 0:
                anomalous_val, anomalous_test = train_test_split(
                    anomalous_trajectories,
                    train_size=0.5,
                    random_state=random_state,
                    shuffle=True
                )
                val_df = pd.concat([val_df, anomalous_val], ignore_index=True)
                test_df = pd.concat([test_df, anomalous_test], ignore_index=True)

            # Shuffle validation and test sets if they have data
            if not val_df.empty:
                val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            if not test_df.empty:
                test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            logger.info("Data splits created (train_ratio ~1.0):")
            logger.info("  Train: %d (%.1f%% normal)", len(train_df), 100.0 if not train_df.empty else 0.0)
            val_normal_perc = 100.0 * (~val_df.get('is_anomalous', pd.Series(dtype=bool))).sum() / len(val_df) if not val_df.empty else 0.0
            logger.info("  Validation: %d (%.1f%% normal)", len(val_df), val_normal_perc)
            test_normal_perc = 100.0 * (~test_df.get('is_anomalous', pd.Series(dtype=bool))).sum() / len(test_df) if not test_df.empty else 0.0
            logger.info("  Test: %d (%.1f%% normal)", len(test_df), test_normal_perc)
            return train_df, val_df, test_df

        # Proceed with standard splitting if train_ratio < 1.0
        train_size = self.train_ratio
        val_test_size = self.val_ratio + self.test_ratio
        if val_test_size == 0: # Should not happen if train_ratio < 1.0 and ratios sum to 1
             logger.error("Validation and Test ratio sum to 0, but train_ratio is less than 1.0. Check data_split configuration.")
             # Fallback: Give all remaining to validation, or handle error appropriately
             # For now, let's prevent ZeroDivisionError by assigning all to normal_val_test to train
             # This case implies a misconfiguration.
             normal_train = normal_trajectories
             normal_val_test = pd.DataFrame(columns=normal_trajectories.columns)

        else:
            val_size_adjusted = self.val_ratio / val_test_size
        
            # First split: train vs (val + test)
            normal_train, normal_val_test = train_test_split(
            normal_trajectories, 
            train_size=train_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: val vs test from the remaining normal data
        normal_val, normal_test = train_test_split(
            normal_val_test,
            train_size=val_size_adjusted,
            random_state=random_state,
            shuffle=True
        )
        
        # Split anomalous trajectories between validation and test
        if len(anomalous_trajectories) > 0:
            anomalous_val, anomalous_test = train_test_split(
                anomalous_trajectories,
                train_size=0.5,  # Split anomalies equally between val and test
                random_state=random_state,
                shuffle=True
            )
        else:
            anomalous_val = pd.DataFrame()
            anomalous_test = pd.DataFrame()
        
        # Create final splits
        train_df = normal_train.copy()  # Only normal trajectories for training
        val_df = pd.concat([normal_val, anomalous_val], ignore_index=True)
        test_df = pd.concat([normal_test, anomalous_test], ignore_index=True)
        
        # Shuffle validation and test sets
        val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        logger.info("Data splits created:")
        logger.info("  Train: %d (%.1f%% normal)", len(train_df), 100.0)
        logger.info("  Validation: %d (%.1f%% normal)", len(val_df), 
                   100.0 * (~val_df.get('is_anomalous', False)).sum() / len(val_df))
        logger.info("  Test: %d (%.1f%% normal)", len(test_df), 
                   100.0 * (~test_df.get('is_anomalous', False)).sum() / len(test_df))
        
        return train_df, val_df, test_df
    
    def calibrate_threshold(self, model_results: Dict[str, Any], val_df: pd.DataFrame, 
                          val_features: np.ndarray) -> Dict[str, float]:
        """
        Calibrate detection threshold using validation set.
        
        Args:
            model_results: Trained model results
            val_df: Validation DataFrame with labels
            val_features: Validation features (without labels)
        
        Returns:
            Dictionary of threshold values for different calibration methods
        """
        logger.info("Calibrating threshold using validation set (%d samples)", len(val_df))
        
        # Get anomaly scores from model
        try:
            from .models import AnomalyDetectionModels
            models_handler = AnomalyDetectionModels(self.config)
            scores, _ = models_handler.predict_anomalies(model_results, val_features)
        except Exception as e:
            logger.error("Failed to get validation scores: %s", e)
            return {'default': 0.0}
        
        # Get true labels (1 for anomalous, 0 for normal)
        y_true = val_df.get('is_anomalous', pd.Series([False] * len(val_df))).astype(int).values
        
        thresholds = {}
        
        # Method 1: ROC curve optimization
        if 'roc_optimization' in self.calibration_methods:
            try:
                fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
                # Find threshold that maximizes TPR - FPR
                optimal_idx = np.argmax(tpr - fpr)
                thresholds['roc_optimization'] = roc_thresholds[optimal_idx]
            except Exception as e:
                logger.warning("ROC optimization failed: %s", e)
                thresholds['roc_optimization'] = np.percentile(scores, 90)
        
        # Method 2: Precision-Recall curve optimization
        if 'pr_optimization' in self.calibration_methods:
            try:
                precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
                # Find threshold that maximizes F1 score
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                if optimal_idx < len(pr_thresholds):
                    thresholds['pr_optimization'] = pr_thresholds[optimal_idx]
                else:
                    thresholds['pr_optimization'] = np.percentile(scores, 90)
            except Exception as e:
                logger.warning("PR optimization failed: %s", e)
                thresholds['pr_optimization'] = np.percentile(scores, 90)
        
        # Method 3: F1 score maximization
        if 'f1_maximization' in self.calibration_methods:
            try:
                best_f1 = 0
                best_threshold = np.percentile(scores, 90)
                
                # Try different percentiles
                for percentile in range(80, 99):
                    threshold = np.percentile(scores, percentile)
                    predictions = (scores >= threshold).astype(int)
                    
                    if np.sum(predictions) > 0:  # Avoid division by zero
                        f1 = f1_score(y_true, predictions)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                thresholds['f1_maximization'] = best_threshold
            except Exception as e:
                logger.warning("F1 maximization failed: %s", e)
                thresholds['f1_maximization'] = np.percentile(scores, 90)
        
        # Method 4: Fixed percentile
        if 'fixed_percentile_95' in self.calibration_methods:
            thresholds['fixed_percentile_95'] = np.percentile(scores, 95)
        
        # Method 5: Knee point detection
        if 'knee_point_detection' in self.calibration_methods:
            try:
                # Use the elbow method on sorted scores
                sorted_scores = np.sort(scores)
                n = len(sorted_scores)
                
                # Calculate differences
                diffs = np.diff(sorted_scores)
                if len(diffs) > 0:
                    # Find the point with maximum acceleration (second derivative)
                    second_diffs = np.diff(diffs)
                    if len(second_diffs) > 0:
                        knee_idx = np.argmax(second_diffs) + 2  # +2 due to double diff
                        if knee_idx < n:
                            thresholds['knee_point_detection'] = sorted_scores[knee_idx]
                        else:
                            thresholds['knee_point_detection'] = np.percentile(scores, 90)
                    else:
                        thresholds['knee_point_detection'] = np.percentile(scores, 90)
                else:
                    thresholds['knee_point_detection'] = np.percentile(scores, 90)
            except Exception as e:
                logger.warning("Knee point detection failed: %s", e)
                thresholds['knee_point_detection'] = np.percentile(scores, 90)
        
        # Default threshold if no methods specified
        if not thresholds:
            thresholds['default'] = np.percentile(scores, 90)
        
        logger.info("Calibrated thresholds: %s", thresholds)
        return thresholds
    
    def evaluate_model(self, model_results: Dict[str, Any], test_df: pd.DataFrame, 
                      test_features: np.ndarray, threshold: float) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            model_results: Trained model results
            test_df: Test DataFrame with labels
            test_features: Test features (without labels)
            threshold: Detection threshold
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set (%d samples)", len(test_df))
        
        # Get predictions from model
        try:
            from .models import AnomalyDetectionModels
            models_handler = AnomalyDetectionModels(self.config)
            scores, predictions = models_handler.predict_anomalies(
                model_results, test_features, threshold
            )
        except Exception as e:
            logger.error("Failed to get test predictions: %s", e)
            return {'error': 1.0}
        
        # Get true labels
        y_true = test_df.get('is_anomalous', pd.Series([False] * len(test_df))).astype(int).values
        
        metrics = {}
        
        try:
            # Basic classification metrics
            if 'precision' in self.metrics:
                if np.sum(predictions) > 0:
                    metrics['precision'] = precision_score(y_true, predictions)
                else:
                    metrics['precision'] = 0.0
            
            if 'recall' in self.metrics:
                if np.sum(y_true) > 0:
                    metrics['recall'] = recall_score(y_true, predictions)
                else:
                    metrics['recall'] = 1.0  # No anomalies to detect
            
            if 'f1' in self.metrics:
                if np.sum(predictions) > 0 and np.sum(y_true) > 0:
                    metrics['f1'] = f1_score(y_true, predictions)
                else:
                    metrics['f1'] = 0.0
            
            if 'accuracy' in self.metrics:
                metrics['accuracy'] = accuracy_score(y_true, predictions)
            
            # AUC metrics (require probabilistic scores)
            if len(set(y_true)) > 1:  # Need both classes for AUC
                if 'auc_roc' in self.metrics:
                    try:
                        metrics['auc_roc'] = roc_auc_score(y_true, scores)
                    except Exception as e:
                        logger.warning("Failed to calculate AUC-ROC: %s", e)
                        metrics['auc_roc'] = 0.5
                
                if 'auc_pr' in self.metrics:
                    try:
                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
                        metrics['auc_pr'] = auc(recall_curve, precision_curve)
                    except Exception as e:
                        logger.warning("Failed to calculate AUC-PR: %s", e)
                        metrics['auc_pr'] = 0.0
            else:
                metrics['auc_roc'] = 0.5
                metrics['auc_pr'] = 0.0
            
            # Clustering metrics (unsupervised quality measures)
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                
                if len(set(predictions)) > 1:  # Need at least 2 clusters
                    if 'silhouette_score' in self.metrics:
                        metrics['silhouette_score'] = silhouette_score(test_features, predictions)
                    
                    if 'calinski_harabasz_score' in self.metrics:
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(test_features, predictions)
                    
                    if 'davies_bouldin_score' in self.metrics:
                        metrics['davies_bouldin_score'] = davies_bouldin_score(test_features, predictions)
                else:
                    metrics['silhouette_score'] = 0.0
                    metrics['calinski_harabasz_score'] = 0.0
                    metrics['davies_bouldin_score'] = np.inf
            except Exception as e:
                logger.warning("Failed to calculate clustering metrics: %s", e)
            
            # Custom anomaly detection metrics
            # Note: User should update their config.yaml if they were using the old metric names.
            if 'overall_detection_recall' in self.metrics: # Renamed from detection_latency
                metrics['overall_detection_recall'] = self._calculate_overall_detection_recall(test_df, y_true, predictions)
            
            if 'false_discovery_rate' in self.metrics: # Renamed from false_positive_clustering
                metrics['false_discovery_rate'] = self._calculate_false_discovery_rate(
                    test_df, y_true, predictions
                )
            
            if 'severity_weighted_performance' in self.metrics:
                metrics['severity_weighted_performance'] = self._calculate_severity_weighted_performance(
                    test_df, y_true, predictions
                )
            
        except Exception as e:
            logger.error("Error calculating metrics: %s", e)
            metrics['error'] = 1.0
        
        logger.info("Evaluation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
        return metrics
    
    def evaluate_by_anomaly_type(self, model_results: Dict[str, Any], test_df: pd.DataFrame, 
                                test_features: np.ndarray, threshold: float) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance by anomaly type.
        
        Args:
            model_results: Trained model results
            test_df: Test DataFrame with labels
            test_features: Test features
            threshold: Detection threshold
        
        Returns:
            Dictionary of metrics by anomaly type
        """
        logger.info("Evaluating by anomaly type")
        
        # Get predictions
        try:
            from .models import AnomalyDetectionModels
            models_handler = AnomalyDetectionModels(self.config)
            scores, predictions = models_handler.predict_anomalies(
                model_results, test_features, threshold
            )
        except Exception as e:
            logger.error("Failed to get predictions: %s", e)
            return {}
        
        # Group by anomaly type
        type_metrics = {}
        
        # Normal trajectories
        normal_mask = ~test_df.get('is_anomalous', pd.Series([False] * len(test_df)))
        if normal_mask.sum() > 0:
            normal_true = np.zeros(normal_mask.sum())
            normal_pred = predictions[normal_mask]
            
            type_metrics['normal'] = {
                'count': normal_mask.sum(),
                'false_positive_rate': normal_pred.sum() / len(normal_pred) if len(normal_pred) > 0 else 0.0,
                'specificity': 1.0 - (normal_pred.sum() / len(normal_pred)) if len(normal_pred) > 0 else 1.0
            }
        
        # Anomalous trajectories by type
        anomalous_df = test_df[test_df.get('is_anomalous', False)]
        
        if len(anomalous_df) > 0:
            # Extract anomaly types (handle both string and list formats)
            anomaly_types = []
            for _, row in anomalous_df.iterrows():
                types = row.get('anomaly_types', [])
                if isinstance(types, str):
                    anomaly_types.append([types])
                elif isinstance(types, list):
                    anomaly_types.append(types)
                else:
                    anomaly_types.append(['unknown'])
            
            # Get unique anomaly types
            all_types = set()
            for type_list in anomaly_types:
                all_types.update(type_list)
            
            for anomaly_type in all_types:
                # Find indices for this anomaly type
                type_indices = []
                for i, type_list in enumerate(anomaly_types):
                    if anomaly_type in type_list:
                        type_indices.append(anomalous_df.index[i])
                
                if type_indices:
                    type_mask = test_df.index.isin(type_indices)
                    type_true = test_df.loc[type_mask, 'is_anomalous'].astype(int).values
                    type_pred = predictions[type_mask]
                    
                    if len(type_true) > 0:
                        type_metrics[anomaly_type] = {
                            'count': len(type_true),
                            'detection_rate': type_pred.sum() / len(type_pred),
                            'recall': recall_score(type_true, type_pred) if type_true.sum() > 0 else 0.0,
                            'precision': precision_score(type_true, type_pred) if type_pred.sum() > 0 else 0.0
                        }
        
        logger.info("Performance by anomaly type: %s", 
                   {k: f"count={v.get('count', 0)}, detection_rate={v.get('detection_rate', 0):.3f}" 
                    for k, v in type_metrics.items()})
        
        return type_metrics
    
    def _calculate_overall_detection_recall(self, test_df: pd.DataFrame, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate overall recall of detected anomalies.
        This was previously named _calculate_detection_latency but its calculation is recall.
        """
        # y_true should be passed to ensure consistency with other metric calculations
        # total_anomalies = test_df.get('is_anomalous', pd.Series([False] * len(test_df))).sum()
        total_anomalies = np.sum(y_true == 1)
        
        # Detected anomalies are true positives (prediction is 1 and true is 1)
        detected_true_anomalies = np.sum((predictions == 1) & (y_true == 1))
        
        if total_anomalies > 0:
            return detected_true_anomalies / total_anomalies
        elif np.sum(predictions == 1) == 0: # No anomalies and no positive predictions
             return 1.0 # Perfect score if there were no anomalies to find and none were predicted
        else: # No true anomalies, but some were predicted
             return 0.0


    def _calculate_false_discovery_rate(self, test_df: pd.DataFrame,
                                           y_true: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate the False Discovery Rate (FDR = FP / (FP + TP) = FP / PRED_P).
        This is 1 - Precision.
        Previously named _calculate_false_positive_clustering.
        """
        false_positives = np.sum((predictions == 1) & (y_true == 0))
        true_positives = np.sum((predictions == 1) & (y_true == 1))
        total_predicted_positives = false_positives + true_positives # Same as np.sum(predictions == 1)
        
        if total_predicted_positives > 0:
            return false_positives / total_predicted_positives
        else: # No positive predictions, so no false discoveries among them.
            return 0.0
    
    def _calculate_severity_weighted_performance(self, test_df: pd.DataFrame, 
                                               y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate performance weighted by anomaly severity."""
        if len(test_df) == 0:
            return 0.0
        
        severity_weights = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, predictions)):
            if i < len(test_df):
                severity = test_df.iloc[i].get('anomaly_severity', 'low')
                weight = severity_weights.get(severity, 1.0)
                
                # Score: 1 for correct prediction, 0 for incorrect
                score = 1.0 if true_label == pred_label else 0.0
                
                # Weight by severity (higher severity errors are more important)
                if true_label == 1:  # Only weight anomalous cases
                    weighted_score += score * weight
                    total_weight += weight
                else:
                    weighted_score += score
                    total_weight += 1.0
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary containing all evaluation results
        
        Returns:
            Structured evaluation report
        """
        report = {
            'summary': {},
            'model_comparison': {},
            'threshold_analysis': {},
            'anomaly_type_analysis': {},
            'recommendations': []
        }
        
        # Summary statistics
        if 'model_results' in results:
            model_results = results['model_results']
            
            best_model = None
            best_f1 = 0.0
            
            for model_name, model_data in model_results.items():
                if 'test_metrics' in model_data:
                    f1 = model_data['test_metrics'].get('f1', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_name
            
            report['summary'] = {
                'best_model': best_model,
                'best_f1_score': best_f1,
                'models_evaluated': len(model_results),
                'total_test_samples': results.get('total_test_samples', 0),
                'anomaly_rate': results.get('anomaly_rate', 0.0)
            }
        
        # Model comparison
        if 'model_results' in results:
            comparison = {}
            for model_name, model_data in results['model_results'].items():
                if 'test_metrics' in model_data:
                    metrics = model_data['test_metrics']
                    comparison[model_name] = {
                        'f1': metrics.get('f1', 0.0),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'auc_roc': metrics.get('auc_roc', 0.5),
                        'auc_pr': metrics.get('auc_pr', 0.0)
                    }
            
            report['model_comparison'] = comparison
        
        # Threshold analysis
        if 'threshold_results' in results:
            threshold_analysis = {}
            for method, metrics in results['threshold_results'].items():
                threshold_analysis[method] = {
                    'f1_score': metrics.get('f1', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0)
                }
            
            report['threshold_analysis'] = threshold_analysis
        
        # Anomaly type analysis
        if 'type_analysis' in results:
            report['anomaly_type_analysis'] = results['type_analysis']
        
        # Generate recommendations
        recommendations = []
        
        if best_model:
            recommendations.append(f"Use {best_model} model for deployment (best F1: {best_f1:.3f})")
        
        # Check for specific issues
        if 'model_results' in results:
            for model_name, model_data in results['model_results'].items():
                metrics = model_data.get('test_metrics', {})
                
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                
                if precision < 0.7:
                    recommendations.append(f"{model_name}: Low precision ({precision:.3f}) - many false positives")
                
                if recall < 0.7:
                    recommendations.append(f"{model_name}: Low recall ({recall:.3f}) - missing anomalies")
        
        report['recommendations'] = recommendations
        
        logger.info("Generated evaluation report with %d recommendations", len(recommendations))
        
        return report
    
    def export_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Export evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Exported evaluation results to %s", filepath)
