#!/usr/bin/env python3
"""
Performance Benchmark Script for Enhanced Anomaly Detection Models.

This script provides:
1. Side-by-side comparison of original vs enhanced models
2. Performance metrics analysis
3. Training time comparisons
4. Memory usage monitoring
5. Detailed performance reports
"""

import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from modules.models import AnomalyDetectionModels
    from modules.enhanced_models import EnhancedAnomalyDetectionModels
    from modules.utils import Timer, ensure_directory
    from modules.data_generation import TrajectoryDataGenerator
    from modules.anomaly_injection import AnomalyInjector
    from modules.graph_processing import GraphProcessor
    from modules.feature_engineering import FeatureExtractor
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install numpy pandas scikit-learn matplotlib seaborn pyyaml torch torch-geometric scikit-optimize")


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for anomaly detection models.
    """
    
    def __init__(self, config_path: str = "config_enhanced.yaml", output_dir: str = "benchmark_results"):
        """Initialize benchmark with configuration."""
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        ensure_directory(str(self.output_dir))
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_generator = TrajectoryDataGenerator(self.config)
        self.anomaly_injector = AnomalyInjector(self.config)
        self.graph_processor = GraphProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        
        # Models
        self.original_models = AnomalyDetectionModels(self.config)
        self.enhanced_models = EnhancedAnomalyDetectionModels(self.config)
        
        # Results storage
        self.results = {
            'original': {},
            'enhanced': {},
            'comparison': {}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark."""
        self.logger.info("Starting comprehensive performance benchmark")
        
        try:
            # 1. Data preparation
            self.logger.info("Step 1: Data preparation")
            data = self._prepare_benchmark_data()
            
            # 2. Model training and evaluation
            self.logger.info("Step 2: Training original models")
            original_results = self._benchmark_original_models(data)
            
            self.logger.info("Step 3: Training enhanced models")
            enhanced_results = self._benchmark_enhanced_models(data)
            
            # 3. Performance comparison
            self.logger.info("Step 4: Performance comparison")
            comparison_results = self._compare_performance(original_results, enhanced_results)
            
            # 4. Generate reports
            self.logger.info("Step 5: Generating reports")
            self._generate_benchmark_report(comparison_results)
            
            return {
                'original': original_results,
                'enhanced': enhanced_results,
                'comparison': comparison_results
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise
    
    def _prepare_benchmark_data(self) -> Dict[str, Any]:
        """Prepare data for benchmarking."""
        self.logger.info("Generating synthetic trajectories")
        
        # Generate normal trajectories
        normal_trajectories = self.data_generator.generate_trajectories(
            num_trajectories=500,  # Reduced for benchmark
            trajectory_type='mixed'
        )
        
        # Inject anomalies
        anomalous_trajectories = self.anomaly_injector.inject_anomalies(
            normal_trajectories[:100],  # Use subset for anomaly injection
            injection_config=self.config.get('anomaly_injection', {})
        )
        
        all_trajectories = normal_trajectories + anomalous_trajectories
        
        # Convert to graphs
        self.logger.info("Converting to graphs")
        graphs = self.graph_processor.convert_trajectories_to_graphs(all_trajectories)
        
        # Extract features
        self.logger.info("Extracting features")
        features_df = self.feature_extractor.extract_features(graphs, all_trajectories)
        
        # Create labels
        labels = [0] * len(normal_trajectories) + [1] * len(anomalous_trajectories)
        features_df['is_anomalous'] = labels
        
        # Data splits
        train_size = int(0.6 * len(features_df))
        val_size = int(0.2 * len(features_df))
        
        train_df = features_df[:train_size]
        val_df = features_df[train_size:train_size + val_size]
        test_df = features_df[train_size + val_size:]
        
        # Ensure training set has only normal trajectories
        train_df = train_df[train_df['is_anomalous'] == 0]
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'graphs': graphs,
            'trajectories': all_trajectories
        }
    
    def _benchmark_original_models(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark original models."""
        self.logger.info("Benchmarking original models")
        
        train_df = data['train_df']
        test_df = data['test_df']
        graphs = data['graphs']
        
        # Prepare training data
        X_train, feature_names = self.original_models.prepare_training_data(train_df)
        X_test, _ = self.original_models.prepare_training_data(
            test_df, feature_columns=feature_names
        )
        y_test = test_df['is_anomalous'].values
        
        results = {}
        
        # Isolation Forest
        self.logger.info("Training original Isolation Forest")
        with Timer() as timer:
            iso_results = self.original_models.train_isolation_forest(
                X_train, hyperparameter_tuning=True
            )
            iso_train_time = timer.elapsed()
        
        # Evaluate
        scores, predictions = self.original_models.predict_anomalies(iso_results, X_test)
        iso_metrics = self._calculate_metrics(y_test, predictions, scores)
        
        results['isolation_forest'] = {
            'training_time': iso_train_time,
            'metrics': iso_metrics,
            'best_params': iso_results.get('best_params', {})
        }
        
        # One-Class SVM
        self.logger.info("Training original One-Class SVM")
        with Timer() as timer:
            svm_results = self.original_models.train_one_class_svm(
                X_train, hyperparameter_tuning=True
            )
            svm_train_time = timer.elapsed()
        
        # Evaluate
        scores, predictions = self.original_models.predict_anomalies(svm_results, X_test)
        svm_metrics = self._calculate_metrics(y_test, predictions, scores)
        
        results['one_class_svm'] = {
            'training_time': svm_train_time,
            'metrics': svm_metrics,
            'best_params': svm_results.get('best_params', {})
        }
        
        # GNN Autoencoder
        self.logger.info("Training original GNN Autoencoder")
        try:
            with Timer() as timer:
                gnn_results = self.original_models.train_gnn_autoencoder(
                    graphs, hyperparameter_tuning=True
                )
                gnn_train_time = timer.elapsed()
            
            # Evaluate with graphs
            test_graphs = graphs[-len(test_df):]  # Approximate test graphs
            scores, predictions = self.original_models.predict_anomalies_graphs(
                gnn_results, test_graphs
            )
            gnn_metrics = self._calculate_metrics(y_test, predictions, scores)
            
            results['gnn_autoencoder'] = {
                'training_time': gnn_train_time,
                'metrics': gnn_metrics,
                'best_params': gnn_results.get('best_params', {})
            }
        except Exception as e:
            self.logger.warning(f"Original GNN training failed: {e}")
            results['gnn_autoencoder'] = {'error': str(e)}
        
        return results
    
    def _benchmark_enhanced_models(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark enhanced models."""
        self.logger.info("Benchmarking enhanced models")
        
        train_df = data['train_df']
        test_df = data['test_df']
        graphs = data['graphs']
        
        # Prepare training data
        X_train, feature_names = self.enhanced_models.prepare_training_data(train_df)
        X_test, _ = self.enhanced_models.prepare_training_data(
            test_df, feature_columns=feature_names
        )
        y_test = test_df['is_anomalous'].values
        
        results = {}
        
        # Enhanced Isolation Forest
        self.logger.info("Training enhanced Isolation Forest")
        with Timer() as timer:
            iso_results = self.enhanced_models.train_enhanced_isolation_forest(
                X_train, hyperparameter_tuning=True
            )
            iso_train_time = timer.elapsed()
        
        # Evaluate ensemble
        if 'models' in iso_results:
            # Ensemble evaluation
            ensemble_scores = np.zeros(len(X_test))
            for model in iso_results['models']:
                ensemble_scores += model.decision_function(X_test)
            ensemble_scores /= len(iso_results['models'])
            
            threshold = np.percentile(ensemble_scores, 90)
            predictions = (ensemble_scores < threshold).astype(int)
            iso_metrics = self._calculate_metrics(y_test, predictions, ensemble_scores)
        else:
            scores, predictions = self._predict_with_model(iso_results, X_test)
            iso_metrics = self._calculate_metrics(y_test, predictions, scores)
        
        results['isolation_forest'] = {
            'training_time': iso_train_time,
            'metrics': iso_metrics,
            'best_params': iso_results.get('best_params', {}),
            'ensemble_size': len(iso_results.get('models', [iso_results.get('model')]))
        }
        
        # Enhanced One-Class SVM
        self.logger.info("Training enhanced One-Class SVM")
        with Timer() as timer:
            svm_results = self.enhanced_models.train_enhanced_one_class_svm(
                X_train, hyperparameter_tuning=True
            )
            svm_train_time = timer.elapsed()
        
        # Evaluate ensemble
        if 'kernel_models' in svm_results:
            # Multi-kernel ensemble evaluation
            if 'weights' in svm_results:
                # Weighted average
                weighted_scores = np.zeros(len(X_test))
                for kernel, weight in svm_results['weights'].items():
                    kernel_scores = svm_results['kernel_models'][kernel]['model'].decision_function(X_test)
                    weighted_scores += weight * kernel_scores
                
                threshold = np.percentile(weighted_scores, 90)
                predictions = (weighted_scores < threshold).astype(int)
                svm_metrics = self._calculate_metrics(y_test, predictions, weighted_scores)
            else:
                # Use best kernel
                best_kernel = list(svm_results['kernel_models'].keys())[0]
                model = svm_results['kernel_models'][best_kernel]['model']
                scores = model.decision_function(X_test)
                predictions = model.predict(X_test)
                predictions = (predictions == -1).astype(int)
                svm_metrics = self._calculate_metrics(y_test, predictions, scores)
        else:
            scores, predictions = self._predict_with_model(svm_results, X_test)
            svm_metrics = self._calculate_metrics(y_test, predictions, scores)
        
        results['one_class_svm'] = {
            'training_time': svm_train_time,
            'metrics': svm_metrics,
            'best_params': svm_results.get('best_params', {}),
            'kernels_used': list(svm_results.get('kernel_models', {}).keys())
        }
        
        # Enhanced GNN Autoencoder
        self.logger.info("Training enhanced GNN Autoencoder")
        try:
            with Timer() as timer:
                gnn_results = self.enhanced_models.train_enhanced_gnn_autoencoder(
                    graphs, hyperparameter_tuning=True
                )
                gnn_train_time = timer.elapsed()
            
            # Evaluate
            test_graphs = graphs[-len(test_df):]
            scores, predictions = self._predict_gnn_anomalies(gnn_results, test_graphs)
            gnn_metrics = self._calculate_metrics(y_test, predictions, scores)
            
            results['gnn_autoencoder'] = {
                'training_time': gnn_train_time,
                'metrics': gnn_metrics,
                'best_params': gnn_results.get('best_params', {}),
                'architecture': gnn_results.get('best_params', {}).get('architecture', 'unknown')
            }
        except Exception as e:
            self.logger.warning(f"Enhanced GNN training failed: {e}")
            results['gnn_autoencoder'] = {'error': str(e)}
        
        return results
    
    def _predict_with_model(self, model_results: Dict[str, Any], X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies with a single model."""
        model = model_results.get('model')
        if model is None:
            return np.zeros(len(X_test)), np.zeros(len(X_test))
        
        try:
            scores = model.decision_function(X_test)
            predictions = model.predict(X_test)
            predictions = (predictions == -1).astype(int)
            return scores, predictions
        except:
            return np.zeros(len(X_test)), np.zeros(len(X_test))
    
    def _predict_gnn_anomalies(self, gnn_results: Dict[str, Any], test_graphs: List) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using GNN model."""
        model = gnn_results.get('model')
        if model is None:
            return np.zeros(len(test_graphs)), np.zeros(len(test_graphs))
        
        try:
            import torch
            from torch_geometric.data import DataLoader
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            model = model.to(device)
            
            scores = []
            with torch.no_grad():
                for graph_data in test_graphs:
                    if hasattr(graph_data, 'x'):
                        graph_data = graph_data.to(device)
                        
                        if hasattr(model, 'encode'):  # Variational
                            reconstructed, mu, logvar = model(graph_data)
                            error = torch.mean((graph_data.x - reconstructed) ** 2).item()
                        else:
                            reconstructed = model(graph_data)
                            error = torch.mean((graph_data.x - reconstructed) ** 2).item()
                        
                        scores.append(error)
                    else:
                        scores.append(0.0)
            
            scores = np.array(scores)
            threshold = np.percentile(scores, 90)
            predictions = (scores > threshold).astype(int)
            
            return scores, predictions
            
        except Exception as e:
            self.logger.warning(f"GNN prediction failed: {e}")
            return np.zeros(len(test_graphs)), np.zeros(len(test_graphs))
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score,
            roc_auc_score, average_precision_score, matthews_corrcoef
        )
        
        metrics = {}
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # AUC metrics (if scores vary)
            if len(np.unique(scores)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, scores)
                metrics['auc_pr'] = average_precision_score(y_true, scores)
            else:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
            
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
            
        except Exception as e:
            self.logger.warning(f"Metric calculation failed: {e}")
            # Return default metrics
            metrics = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'auc_roc': 0.0, 'auc_pr': 0.0, 'mcc': 0.0
            }
        
        return metrics
    
    def _compare_performance(self, original_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between original and enhanced models."""
        comparison = {}
        
        for model_name in ['isolation_forest', 'one_class_svm', 'gnn_autoencoder']:
            if model_name in original_results and model_name in enhanced_results:
                orig = original_results[model_name]
                enh = enhanced_results[model_name]
                
                if 'error' not in orig and 'error' not in enh:
                    comparison[model_name] = {
                        'training_time_improvement': self._calculate_improvement(
                            orig['training_time'], enh['training_time'], lower_is_better=True
                        ),
                        'f1_improvement': self._calculate_improvement(
                            orig['metrics']['f1'], enh['metrics']['f1']
                        ),
                        'precision_improvement': self._calculate_improvement(
                            orig['metrics']['precision'], enh['metrics']['precision']
                        ),
                        'recall_improvement': self._calculate_improvement(
                            orig['metrics']['recall'], enh['metrics']['recall']
                        ),
                        'auc_roc_improvement': self._calculate_improvement(
                            orig['metrics']['auc_roc'], enh['metrics']['auc_roc']
                        ),
                        'original_metrics': orig['metrics'],
                        'enhanced_metrics': enh['metrics']
                    }
        
        return comparison
    
    def _calculate_improvement(self, original: float, enhanced: float, lower_is_better: bool = False) -> float:
        """Calculate percentage improvement."""
        if original == 0:
            return 100.0 if enhanced > 0 else 0.0
        
        improvement = ((enhanced - original) / original) * 100
        
        if lower_is_better:
            improvement = -improvement
        
        return improvement
    
    def _generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report."""
        self.logger.info("Generating benchmark report")
        
        # Text report
        report_lines = []
        report_lines.append("# Performance Benchmark Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("## Performance Comparison Summary")
        report_lines.append("")
        report_lines.append("| Model | F1 Improvement | Precision Improvement | Recall Improvement | AUC-ROC Improvement | Training Time Change |")
        report_lines.append("|-------|----------------|----------------------|-------------------|---------------------|---------------------|")
        
        for model_name, comparison in results.get('comparison', {}).items():
            f1_imp = comparison.get('f1_improvement', 0.0)
            prec_imp = comparison.get('precision_improvement', 0.0)
            rec_imp = comparison.get('recall_improvement', 0.0)
            auc_imp = comparison.get('auc_roc_improvement', 0.0)
            time_imp = comparison.get('training_time_improvement', 0.0)
            
            report_lines.append(
                f"| {model_name} | {f1_imp:+.1f}% | {prec_imp:+.1f}% | {rec_imp:+.1f}% | {auc_imp:+.1f}% | {time_imp:+.1f}% |"
            )
        
        report_lines.append("")
        
        # Detailed results
        report_lines.append("## Detailed Results")
        report_lines.append("")
        
        for model_name in ['isolation_forest', 'one_class_svm', 'gnn_autoencoder']:
            if model_name in results.get('comparison', {}):
                comparison = results['comparison'][model_name]
                
                report_lines.append(f"### {model_name.replace('_', ' ').title()}")
                report_lines.append("")
                
                # Original metrics
                orig_metrics = comparison['original_metrics']
                report_lines.append("**Original Model:**")
                report_lines.append(f"- F1 Score: {orig_metrics['f1']:.3f}")
                report_lines.append(f"- Precision: {orig_metrics['precision']:.3f}")
                report_lines.append(f"- Recall: {orig_metrics['recall']:.3f}")
                report_lines.append(f"- AUC-ROC: {orig_metrics['auc_roc']:.3f}")
                report_lines.append("")
                
                # Enhanced metrics
                enh_metrics = comparison['enhanced_metrics']
                report_lines.append("**Enhanced Model:**")
                report_lines.append(f"- F1 Score: {enh_metrics['f1']:.3f}")
                report_lines.append(f"- Precision: {enh_metrics['precision']:.3f}")
                report_lines.append(f"- Recall: {enh_metrics['recall']:.3f}")
                report_lines.append(f"- AUC-ROC: {enh_metrics['auc_roc']:.3f}")
                report_lines.append("")
        
        # Save report
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save JSON results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(results)
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate performance visualization charts."""
        try:
            plt.style.use('seaborn-v0_8')
            
            # Performance comparison chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            models = list(results.get('comparison', {}).keys())
            metrics = ['f1_improvement', 'precision_improvement', 'recall_improvement', 'auc_roc_improvement']
            metric_names = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i // 2, i % 2]
                
                improvements = [results['comparison'][model].get(metric, 0.0) for model in models]
                
                bars = ax.bar(range(len(models)), improvements, 
                             color=['green' if x > 0 else 'red' for x in improvements])
                
                ax.set_title(f'{name} Improvement (%)')
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, improvements):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                           f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Benchmark visualizations saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")


def main():
    """Main function for running the benchmark."""
    parser = argparse.ArgumentParser(description='Performance Benchmark for Enhanced Anomaly Detection Models')
    parser.add_argument('--config', default='config_enhanced.yaml', help='Configuration file path')
    parser.add_argument('--output', default='benchmark_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with reduced data')
    
    args = parser.parse_args()
    
    # Quick benchmark modifications
    if args.quick:
        print("Running quick benchmark...")
        # Modify config for faster execution
        quick_config = {
            'data_generation': {'num_normal_trajectories': 100},
            'models': {
                'isolation_forest': {'n_iterations': 10},
                'one_class_svm': {'n_iterations': 10},
                'gnn_autoencoder': {'n_iterations': 5}
            }
        }
        
        # Load and modify config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply quick modifications
        for key, value in quick_config.items():
            if key in config:
                config[key].update(value)
        
        # Save modified config
        quick_config_path = 'config_quick_benchmark.yaml'
        with open(quick_config_path, 'w') as f:
            yaml.dump(config, f)
        
        args.config = quick_config_path
    
    # Run benchmark
    benchmark = PerformanceBenchmark(args.config, args.output)
    results = benchmark.run_full_benchmark()
    
    print(f"\nBenchmark completed! Results saved to: {args.output}")
    print(f"View the report: {args.output}/benchmark_report.md")
    
    return results


if __name__ == "__main__":
    main()