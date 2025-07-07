#!/usr/bin/env python3
"""
Main orchestration script for AI Agent Trajectory Anomaly Detection System.

Coordinates the complete pipeline:
1. Configuration loading and validation
2. Synthetic data generation
3. Anomaly injection
4. Graph processing and embedding generation
5. Feature extraction and engineering
6. Model training with hyperparameter tuning
7. Threshold calibration
8. Comprehensive evaluation
9. Visualization and reporting
10. Model and result persistence

Usage:
    python main.py [--config config.yaml] [--output-dir results/] [--debug]
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Core pipeline modules
from modules.anomaly_injection import AnomalyInjector
from modules.confidence_analysis import ConfidenceAnalyzer
from modules.data_generation import TrajectoryGenerator
from modules.evaluation import AnomalyDetectionEvaluator
from modules.feature_engineering import FeatureExtractor
from modules.graph_processing import GraphProcessor
from modules.models import AnomalyDetectionModels
from modules.utils import Timer, ensure_directory, load_config, save_pickle, setup_logging

# Visualization modules
from visualisation.visualization import AnomalyDetectionVisualizer
from visualisation.trajectory_visualization import TrajectoryVisualizer
from visualisation.model_performance_visualization import ModelPerformanceVisualizer
from visualisation.embedding_visualization import EmbeddingVisualizer
from visualisation.anomaly_data_visualization import AnomalyDataVisualizer


class AnomalyDetectionPipeline:
    """
    Complete pipeline for AI agent trajectory anomaly detection.
    Orchestrates the workflow from data generation through model evaluation and visualization.
    """
    def __init__(self, config_path: str = "config.yaml", output_dir: str = "results"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self._setup_directories()
        self.config = load_config(config_path)
        self.logger = setup_logging(
            log_level=self.config.get('system', {}).get('logging', {}).get('level', 'INFO'),
            log_file=str(self.output_dir / "logs" / "anomaly_detection.log")
        )
        # Initialize pipeline components
        self.trajectory_generator = TrajectoryGenerator(self.config, self.config.get('system', {}).get('random_seed', 42))
        self.anomaly_injector = AnomalyInjector(self.config, self.config.get('system', {}).get('random_seed', 42))
        self.graph_processor = GraphProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.models = AnomalyDetectionModels(self.config)
        self.evaluator = AnomalyDetectionEvaluator(self.config)
        self.visualizer = AnomalyDetectionVisualizer(self.config, str(self.output_dir / "charts"))
        self.confidence_analyzer = ConfidenceAnalyzer(self.config, str(self.output_dir / "charts"))
        self.trajectory_visualizer = TrajectoryVisualizer(self.config, str(self.output_dir / "charts"))
        self.model_performance_visualizer = ModelPerformanceVisualizer(self.config, str(self.output_dir / "charts"))
        self.embedding_visualizer = EmbeddingVisualizer(self.config, str(self.output_dir / "charts"))
        self.anomaly_data_visualizer = AnomalyDataVisualizer(self.config, str(self.output_dir / "charts"))
        self.results = {'pipeline_start_time': datetime.now().isoformat(), 'config': self.config, 'output_dir': str(self.output_dir)}
        self.logger.info("AnomalyDetectionPipeline initialized")
        self.logger.info("Output directory: %s", self.output_dir)

    def _setup_directories(self):
        for subdir in ["data", "models", "charts", "logs", "reports"]:
            ensure_directory(str(self.output_dir / subdir))

    def run_complete_pipeline(self) -> Dict[str, Any]:
        self.logger.info("=" * 80)
        self.logger.info("STARTING AI AGENT TRAJECTORY ANOMALY DETECTION PIPELINE")
        self.logger.info("=" * 80)
        pipeline_timer = Timer().start()
        try:
            normal_trajectories, anomalous_trajectories = self._generate_synthetic_data()
            all_trajectories = normal_trajectories + anomalous_trajectories
            graphs = self._convert_to_graphs(all_trajectories)
            embeddings = self._generate_embeddings(graphs)
            features_df = self._extract_features(graphs, all_trajectories)
            train_df, val_df, test_df = self._create_data_splits(features_df)
            model_results = self._train_models(train_df, embeddings)
            threshold_results = self._calibrate_thresholds(model_results, val_df, test_df)
            evaluation_results = self._evaluate_models(model_results, test_df, threshold_results)
            visualization_results = self._generate_visualizations(
                graphs, features_df, model_results, evaluation_results, normal_trajectories, anomalous_trajectories
            )
            self._save_results_and_report(model_results, evaluation_results, visualization_results)
            pipeline_timer.stop()
            self.results.update({
                'pipeline_end_time': datetime.now().isoformat(),
                'total_pipeline_time': pipeline_timer.elapsed(),
                'normal_trajectories_count': len(normal_trajectories),
                'anomalous_trajectories_count': len(anomalous_trajectories),
                'total_trajectories': len(all_trajectories),
                'features_count': len(features_df.columns) - 3,
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'visualization_results': visualization_results,
                'success': True
            })
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("Total execution time: %.2f seconds", pipeline_timer.elapsed())
            self.logger.info("Results saved to: %s", self.output_dir)
            self.logger.info("=" * 80)
            return self.results
        except Exception as e:
            pipeline_timer.stop()
            self.logger.error("PIPELINE FAILED: %s", str(e))
            self.logger.error("Traceback: %s", traceback.format_exc())
            self.results.update({
                'pipeline_end_time': datetime.now().isoformat(),
                'total_pipeline_time': pipeline_timer.elapsed(),
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            })
            return self.results
    
    def _generate_synthetic_data(self) -> tuple:
        """Generate synthetic normal and anomalous trajectories."""
        with Timer() as timer:
            # Generate normal trajectories
            num_normal = self.config.get('data_generation', {}).get('num_normal_trajectories', 1000)
            normal_trajectories = self.trajectory_generator.generate_trajectories(num_normal)
            
            # Generate anomalous trajectories by injecting anomalies
            num_anomalous = self.config.get('anomaly_injection', {}).get('total_anomalous_trajectories', 200)
            anomalous_trajectories = self.anomaly_injector.inject_anomalies(
                normal_trajectories, num_anomalous
            )
            
            # Save trajectories
            save_pickle(normal_trajectories, str(self.output_dir / "data" / "normal_trajectories.pkl"))
            save_pickle(anomalous_trajectories, str(self.output_dir / "data" / "anomalous_trajectories.pkl"))
            
            # Log statistics
            anomaly_stats = self.anomaly_injector.get_anomaly_statistics(anomalous_trajectories)
            self.logger.info("Generated %d normal trajectories", len(normal_trajectories))
            self.logger.info("Generated %d anomalous trajectories", len(anomalous_trajectories))
            self.logger.info("Anomaly statistics: %s", anomaly_stats)
            
            self.results['data_generation'] = {
                'normal_count': len(normal_trajectories),
                'anomalous_count': len(anomalous_trajectories),
                'generation_time': timer.elapsed(),
                'anomaly_statistics': anomaly_stats
            }
            
            return normal_trajectories, anomalous_trajectories
    
    def _convert_to_graphs(self, trajectories: List) -> List:
        """Convert trajectories to NetworkX graphs."""
        with Timer() as timer:
            graphs = self.graph_processor.trajectories_to_graphs(trajectories)
            
            # Save graphs
            save_pickle(graphs, str(self.output_dir / "data" / "trajectory_graphs.pkl"))
            
            # Get graph statistics
            graph_stats = self.graph_processor.get_graph_statistics(graphs)
            # Log simplified graph statistics
            simplified_stats = {}
            for k, v in graph_stats.items():
                if k not in ['node_types', 'edge_types']:
                    if isinstance(v, dict) and 'mean' in v:
                        simplified_stats[k] = f"{v['mean']:.2f}"
                    elif isinstance(v, (int, float)):
                        simplified_stats[k] = f"{v:.2f}" if isinstance(v, float) else v
                    else:
                        simplified_stats[k] = v
            
            self.logger.info("Graph statistics: %s", simplified_stats)
            
            self.results['graph_conversion'] = {
                'graph_count': len(graphs),
                'conversion_time': timer.elapsed(),
                'graph_statistics': graph_stats
            }
            
            return graphs
    
    def _generate_embeddings(self, graphs: List) -> Dict[str, Any]:
        """Generate graph embeddings using multiple methods."""
        embeddings_results = {}
        
        # Node2Vec embeddings
        try:
            with Timer() as timer:
                node2vec_results = self.graph_processor.generate_node2vec_embeddings(
                    graphs, hyperparameter_tuning=True
                )
                embeddings_results['node2vec'] = node2vec_results
                embeddings_results['node2vec']['generation_time'] = timer.elapsed()
                
                self.logger.info("Node2Vec embeddings generated in %.2f seconds", timer.elapsed())
                self.logger.info("Best Node2Vec parameters: %s", node2vec_results.get('best_params', {}))
        except Exception as e:
            self.logger.warning("Node2Vec embedding generation failed: %s", e)
            embeddings_results['node2vec'] = {'error': str(e)}
        
        # DeepWalk embeddings
        try:
            with Timer() as timer:
                deepwalk_results = self.graph_processor.generate_deepwalk_embeddings(
                    graphs, hyperparameter_tuning=True
                )
                embeddings_results['deepwalk'] = deepwalk_results
                embeddings_results['deepwalk']['generation_time'] = timer.elapsed()
                
                self.logger.info("DeepWalk embeddings generated in %.2f seconds", timer.elapsed())
                self.logger.info("Best DeepWalk parameters: %s", deepwalk_results.get('best_params', {}))
        except Exception as e:
            self.logger.warning("DeepWalk embedding generation failed: %s", e)
            embeddings_results['deepwalk'] = {'error': str(e)}
        
        # GraphSAGE embeddings (primary method)
        try:
            with Timer() as timer:
                graphsage_results = self.graph_processor.generate_graphsage_embeddings(
                    graphs, hyperparameter_tuning=True
                )
                embeddings_results['graphsage'] = graphsage_results
                embeddings_results['graphsage']['generation_time'] = timer.elapsed()
                
                self.logger.info("GraphSAGE embeddings generated in %.2f seconds", timer.elapsed())
                self.logger.info("Best GraphSAGE parameters: %s", graphsage_results.get('best_params', {}))
        except Exception as e:
            self.logger.warning("GraphSAGE embedding generation failed: %s", e)
            embeddings_results['graphsage'] = {'error': str(e)}
        
        # Save embeddings
        save_pickle(embeddings_results, str(self.output_dir / "data" / "graph_embeddings.pkl"))
        
        self.results['embeddings'] = embeddings_results
        
        return embeddings_results
    
    def _extract_features(self, graphs: List, trajectories: List) -> pd.DataFrame:
        """Extract comprehensive features from graphs."""
        with Timer() as timer:
            features_df = self.feature_extractor.extract_features(graphs)
            
            # Add trajectory metadata (including anomaly labels for evaluation)
            trajectory_metadata = []
            for trajectory in trajectories:
                metadata = {
                    'is_anomalous': trajectory.is_anomalous,
                    'anomaly_severity': trajectory.anomaly_severity.value if trajectory.anomaly_severity else None,
                    'anomaly_types': [at.value for at in trajectory.anomaly_types] if trajectory.anomaly_types else []
                }
                trajectory_metadata.append(metadata)
            
            metadata_df = pd.DataFrame(trajectory_metadata, index=features_df.index)
            features_df = pd.concat([features_df, metadata_df], axis=1)
            
            # Save features
            features_df.to_csv(str(self.output_dir / "data" / "extracted_features.csv"), index=False)
            save_pickle(features_df, str(self.output_dir / "data" / "extracted_features.pkl"))
            
            # Feature analysis
            feature_analysis = self.feature_extractor.get_feature_importance_analysis(features_df)
            
            self.logger.info("Extracted %d features from %d trajectories", 
                           len(features_df.columns) - 3, len(features_df))  # -3 for metadata
            self.logger.info("Feature extraction completed in %.2f seconds", timer.elapsed())
            
            self.results['feature_extraction'] = {
                'feature_count': len(features_df.columns) - 3,
                'trajectory_count': len(features_df),
                'extraction_time': timer.elapsed(),
                'feature_analysis': feature_analysis
            }
            
            return features_df
    
    def _create_data_splits(self, features_df: pd.DataFrame) -> tuple:
        """Create proper unsupervised data splits."""
        with Timer() as timer:
            train_df, val_df, test_df = self.evaluator.create_unsupervised_data_splits(
                features_df, random_state=self.config.get('system', {}).get('random_seed', 42)
            )
            
            # Save splits
            train_df.to_csv(str(self.output_dir / "data" / "train_split.csv"), index=False)
            val_df.to_csv(str(self.output_dir / "data" / "val_split.csv"), index=False)
            test_df.to_csv(str(self.output_dir / "data" / "test_split.csv"), index=False)
            
            self.logger.info("Data splits created in %.2f seconds", timer.elapsed())
            
            self.results['data_splits'] = {
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'split_time': timer.elapsed(),
                'train_anomaly_rate': 0.0,  # Training set is purely normal
                'val_anomaly_rate': val_df.get('is_anomalous', pd.Series([False] * len(val_df))).mean(),
                'test_anomaly_rate': test_df.get('is_anomalous', pd.Series([False] * len(test_df))).mean()
            }
            
            return train_df, val_df, test_df
    
    def _train_models(self, train_df: pd.DataFrame, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Train all anomaly detection models (simplified, less repetitive)."""
        model_results = {}
        best_embeddings = self._select_best_embeddings(embeddings)
        X_train, feature_names = self.models.prepare_training_data(train_df, best_embeddings)
        self.train_feature_columns = feature_names
        self.logger.info("Training data shape: %s", X_train.shape)
        self.logger.info("Training on purely normal trajectories (unsupervised approach)")
        
        # Load all graphs and filter for training set
        all_graphs = []
        train_graphs = []
        if (self.output_dir / "data" / "trajectory_graphs.pkl").exists():
            all_graphs = self.graph_processor.load_graphs(str(self.output_dir / "data" / "trajectory_graphs.pkl"))
            
            # Filter graphs to only include those in the training set
            if 'graph_id' in train_df.columns:
                # Map train_df graph_ids to graph indices
                train_graph_ids = set(train_df['graph_id'].tolist())
                train_graphs = []
                for i, graph in enumerate(all_graphs):
                    graph_id = f"graph_{i}"
                    if graph_id in train_graph_ids:
                        train_graphs.append(graph)
                self.logger.info("Filtered %d graphs for GNN training from %d total graphs", 
                               len(train_graphs), len(all_graphs))
            else:
                # Fallback: assume train_df order matches graph order
                train_graphs = all_graphs[:len(train_df)]
                self.logger.info("Using first %d graphs for GNN training (fallback method)", len(train_graphs))
        
        # Define model training functions and names
        model_trainers = [
            ("isolation_forest", self.models.train_isolation_forest),
            ("one_class_svm", self.models.train_one_class_svm),
            ("gnn_autoencoder", lambda X, _: self.models.train_gnn_autoencoder(train_graphs, True) if train_graphs else {"error": "No training graphs available"})
        ]
        
        for name, trainer in model_trainers:
            try:
                with Timer() as timer:
                    if name == "gnn_autoencoder":
                        result = trainer(X_train, feature_names)
                    else:
                        result = trainer(X_train, True)
                    result['training_time'] = timer.elapsed()
                    result['feature_names'] = feature_names
                    model_results[name] = result
                    self.logger.info(f"{name.replace('_', ' ').title()} trained in %.2f seconds", timer.elapsed())
                    if 'best_params' in result:
                        self.logger.info("Best parameters: %s", result.get('best_params', {}))
            except Exception as e:
                self.logger.error(f"{name.replace('_', ' ').title()} training failed: %s", e)
                model_results[name] = {'error': str(e)}
        
        # Ensemble model (if at least 2 base models succeeded)
        successful_models = {k: v for k, v in model_results.items() if 'error' not in v}
        if len(successful_models) >= 2:
            try:
                val_size = min(100, len(X_train) // 5)
                val_indices = np.random.choice(len(X_train), val_size, replace=False)
                val_features = X_train[val_indices]
                val_labels = np.zeros(len(val_features))
                with Timer() as timer:
                    ensemble_results = self.models.train_ensemble_model(successful_models, val_features, val_labels)
                    if ensemble_results:
                        ensemble_results['training_time'] = timer.elapsed()
                        ensemble_results['feature_names'] = feature_names
                        model_results['ensemble_model'] = ensemble_results
                        self.logger.info("Ensemble model trained in %.2f seconds", timer.elapsed())
                        if 'weights' in ensemble_results:
                            self.logger.info("Ensemble weights: %s", ensemble_results['weights'])
            except Exception as e:
                self.logger.error("Ensemble model training failed: %s", e)
                model_results['ensemble_model'] = {'error': str(e)}
        else:
            self.logger.info("Skipping ensemble training - need at least 2 successful base models (have %d)", len(successful_models))
        
        ensure_directory(str(self.output_dir / "models"))
        self.models.save_models(model_results, str(self.output_dir / "models" / "trained_models_complete.pkl"))
        self.results['model_training'] = {
            'models_trained': len([k for k, v in model_results.items() if 'error' not in v]),
            'models_failed': len([k for k, v in model_results.items() if 'error' in v]),
            'training_features': X_train.shape[1],
            'training_samples': X_train.shape[0],
            'gnn_training_graphs': len(train_graphs) if train_graphs else 0
        }
        return model_results
    
    def _select_best_embeddings(self, embeddings: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """Select the best embeddings based on quality scores, prioritizing GraphSAGE."""
        best_embeddings = None
        best_score = -np.inf
        best_method = None
        
        # Priority order: GraphSAGE > Node2Vec > DeepWalk
        priority_methods = ['graphsage', 'node2vec', 'deepwalk']
        
        for method in priority_methods:
            if method in embeddings:
                embedding_data = embeddings[method]
                if 'embeddings' in embedding_data and 'best_score' in embedding_data:
                    score = embedding_data['best_score']
                    if score > best_score:
                        best_score = score
                        best_embeddings = embedding_data['embeddings']
                        best_method = method
        
        # If no priority methods worked, try any available method
        if best_embeddings is None:
            for method, embedding_data in embeddings.items():
                if 'embeddings' in embedding_data and 'best_score' in embedding_data:
                    score = embedding_data['best_score']
                    if score > best_score:
                        best_score = score
                        best_embeddings = embedding_data['embeddings']
                        best_method = method
        
        if best_embeddings:
            self.logger.info("Selected %s embeddings with score: %.4f", best_method, best_score)
        
        return best_embeddings
    
    def _calibrate_thresholds(self, model_results: Dict[str, Any], val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate detection thresholds using validation set, then evaluate each threshold on the test set."""
        threshold_results = {}
        val_features, _ = self.models.prepare_training_data(val_df, feature_columns=getattr(self, 'train_feature_columns', None))
        test_features, _ = self.models.prepare_training_data(test_df, feature_columns=getattr(self, 'train_feature_columns', None))
        for model_name, model_data in model_results.items():
            if 'error' not in model_data:
                try:
                    with Timer() as timer:
                        thresholds = self.evaluator.calibrate_threshold(
                            model_data, val_df, val_features
                        )
                        method_results = {}
                        for method, threshold in thresholds.items():
                            val_metrics = self.evaluator.evaluate_model(
                                model_data, val_df, val_features, threshold
                            )
                            test_metrics = self.evaluator.evaluate_model(
                                model_data, test_df, test_features, threshold
                            )
                            method_results[method] = {
                                'threshold': threshold,
                                'val_metrics': val_metrics,
                                'test_metrics': test_metrics,
                                'calibration_time': timer.elapsed()
                            }
                        threshold_results[model_name] = method_results
                        self.logger.info("Thresholds calibrated for %s in %.2f seconds", 
                                       model_name, timer.elapsed())
                except Exception as e:
                    self.logger.error("Threshold calibration failed for %s: %s", model_name, e)
                    threshold_results[model_name] = {'error': str(e)}
        return threshold_results
    
    def _evaluate_models(self, model_results: Dict[str, Any], test_df: pd.DataFrame, 
                        threshold_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all models on test set."""
        evaluation_results = {}
        
        # Prepare test data
        test_features, _ = self.models.prepare_training_data(test_df, feature_columns=getattr(self, 'train_feature_columns', None))
        # Load test graphs for GNN
        graphs_file = self.output_dir / "data" / "trajectory_graphs.pkl"
        test_graphs = None
        if graphs_file.exists():
            all_graphs = self.graph_processor.load_graphs(str(graphs_file))
            # Map test_df rows to graphs using 'graph_id' column
            if 'graph_id' in test_df.columns:
                test_graphs = []
                graph_id_to_graph = {f"graph_{i}": g for i, g in enumerate(all_graphs)}
                for gid in test_df['graph_id']:
                    test_graphs.append(graph_id_to_graph.get(gid))
            else:
                # Fallback: assume order matches
                test_graphs = all_graphs[-len(test_df):]
        
        for model_name, model_data in model_results.items():
            if 'error' not in model_data:
                try:
                    model_evaluation = {}
                    
                    # Select best threshold method
                    best_threshold = 0.0
                    best_method = 'default'
                    
                    if model_name in threshold_results:
                        best_f1 = 0.0
                        for method, method_data in threshold_results[model_name].items():
                            if 'metrics' in method_data:
                                f1 = method_data['metrics'].get('f1', 0.0)
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_threshold = method_data['threshold']
                                    best_method = method
                    
                    # Evaluate with best threshold
                    with Timer() as timer:
                        # Use graph-to-graph prediction for GNN
                        if model_name == 'gnn_autoencoder' and test_graphs is not None:
                            test_metrics = self.evaluator.evaluate_model(
                                model_data, test_df, test_graphs, best_threshold
                            )
                            type_analysis = self.evaluator.evaluate_by_anomaly_type(
                                model_data, test_df, test_graphs, best_threshold
                            )
                        else:
                            test_metrics = self.evaluator.evaluate_model(
                                model_data, test_df, test_features, best_threshold
                            )
                            type_analysis = self.evaluator.evaluate_by_anomaly_type(
                                model_data, test_df, test_features, best_threshold
                            )
                        model_evaluation = {
                            'test_metrics': test_metrics,
                            'type_analysis': type_analysis,
                            'best_threshold': best_threshold,
                            'best_threshold_method': best_method,
                            'evaluation_time': timer.elapsed()
                        }
                        evaluation_results[model_name] = model_evaluation
                        self.logger.info("Evaluated %s - F1: %.3f, Precision: %.3f, Recall: %.3f", 
                                       model_name, 
                                       test_metrics.get('f1', 0.0),
                                       test_metrics.get('precision', 0.0),
                                       test_metrics.get('recall', 0.0))
                except Exception as e:
                    self.logger.error("Evaluation failed for %s: %s", model_name, e)
                    evaluation_results[model_name] = {'error': str(e)}
        
        # Generate evaluation report
        try:
            combined_results = {
                'model_results': {k: v for k, v in model_results.items() 
                                if 'error' not in v and k in evaluation_results},
                'threshold_results': threshold_results,
                'type_analysis': {k: v.get('type_analysis', {}) for k, v in evaluation_results.items()},
                'total_test_samples': len(test_df),
                'anomaly_rate': test_df.get('is_anomalous', pd.Series([False] * len(test_df))).mean()
            }
            
            # Update model results with test metrics
            for model_name in combined_results['model_results']:
                if model_name in evaluation_results:
                    combined_results['model_results'][model_name]['test_metrics'] = evaluation_results[model_name].get('test_metrics', {})
            
            evaluation_report = self.evaluator.generate_evaluation_report(combined_results)
            
            # Save evaluation results
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            self.evaluator.export_results(
                evaluation_report, 
                str(reports_dir / "evaluation_report.json")
            )
            
            evaluation_results['evaluation_report'] = evaluation_report
            
        except Exception as e:
            self.logger.error("Evaluation report generation failed: %s", e)
            evaluation_results['report_error'] = str(e)
        
        return evaluation_results
    
    def _generate_visualizations(self, graphs, features_df, model_results, evaluation_results, normal_trajectories, anomalous_trajectories):
        """Generate all visualizations in a simple, loop-based way, including individual trajectory charts."""
        visualization_results = {}
        test_df = pd.read_csv(self.output_dir / "data" / "test_split.csv")
        test_features, _ = self.models.prepare_training_data(test_df)
        test_features_for_confidence, _ = self.models.prepare_training_data(test_df, feature_columns=self.train_feature_columns)
        viz_tasks = [
            ("embedding_visualization", lambda: self.embedding_visualizer.plot_tsne_embeddings(embeddings, features_df['is_anomalous'].astype(int).values) if (embeddings := self._select_best_embeddings(self.results.get('embeddings', {}))) is not None and 'is_anomalous' in features_df.columns else None),
            ("trajectory_examples", lambda: self.trajectory_visualizer.plot_trajectory_examples(graphs, list(range(len(normal_trajectories))), [i + len(normal_trajectories) for i in range(len(anomalous_trajectories))])),
            ("performance_comparison", lambda: self.model_performance_visualizer.plot_model_performance_comparison(model_results)),
            ("roc_curves", lambda: self.visualizer.plot_roc_curves(model_results, test_df, test_features)),
            ("pr_curves", lambda: self.visualizer.plot_precision_recall_curves(model_results, test_df, test_features)),
            ("feature_importance", lambda: self.visualizer.plot_feature_importance(features_df)),
            ("anomaly_distribution", lambda: self.anomaly_data_visualizer.plot_anomaly_distribution(test_df)),
            ("ensemble_weights", lambda: self.visualizer.plot_ensemble_weights(model_results['ensemble_model']) if 'ensemble_model' in model_results and 'error' not in model_results['ensemble_model'] else None),
            ("ensemble_performance", lambda: self.visualizer.plot_ensemble_performance_comparison(model_results) if 'ensemble_model' in model_results and 'error' not in model_results['ensemble_model'] else None),
            ("confidence_distributions", lambda: self.confidence_analyzer.plot_confidence_distributions(model_results, test_df, test_features_for_confidence)),
            ("confidence_vs_performance", lambda: self.confidence_analyzer.plot_confidence_vs_performance(model_results, test_df, test_features_for_confidence)),
            ("calibration_analysis", lambda: self.confidence_analyzer.plot_calibration_analysis(model_results, test_df, test_features_for_confidence)),
            ("confidence_error_analysis", lambda: self.confidence_analyzer.plot_confidence_error_analysis(model_results, test_df, test_features_for_confidence)),
            ("comprehensive_confidence_analysis", lambda: self.confidence_analyzer.generate_comprehensive_confidence_report(model_results, test_df, test_features_for_confidence)),
            ("summary_report", lambda: self.visualizer.generate_summary_report(self.results)),
        ]
        for name, func in viz_tasks:
            try:
                with Timer() as timer:
                    result = func()
                    if result is not None:
                        visualization_results[name] = {'file': result, 'generation_time': timer.elapsed()}
                        self.logger.info(f"Generated {name.replace('_', ' ')}")
            except Exception as e:
                self.logger.warning(f"{name.replace('_', ' ').title()} failed: %s", e)
        # --- New: Generate individual trajectory charts ---
        try:
            traj_dir = self.output_dir / "charts" / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            # Only plot a few normal and a few anomalous trajectories
            num_examples = 3
            selected_normals = normal_trajectories[:num_examples]
            selected_anomalies = anomalous_trajectories[:num_examples]
            for i, traj in enumerate(selected_normals):
                out_name = f"normal_trajectory_{i+1}"
                self.trajectory_visualizer.plot_enhanced_trajectory(traj, output_path=out_name)
            for i, traj in enumerate(selected_anomalies):
                out_name = f"anomalous_trajectory_{i+1}"
                self.trajectory_visualizer.plot_enhanced_trajectory(traj, output_path=out_name)
            visualization_results["individual_trajectories"] = {
                "directory": str(traj_dir),
                "normal_count": len(selected_normals),
                "anomalous_count": len(selected_anomalies)
            }
            self.logger.info(f"Generated {len(selected_normals)} normal and {len(selected_anomalies)} anomalous trajectory charts in {traj_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to generate individual trajectory charts: {e}")
        return visualization_results
    
    def _save_results_and_report(self, model_results: Dict[str, Any], 
                               evaluation_results: Dict[str, Any], 
                               visualization_results: Dict[str, Any]) -> None:
        """Save final results and generate comprehensive report."""
        try:
            # Ensure reports directory exists
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Save complete results
            complete_results = {
                'pipeline_results': self.results,
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'visualization_results': visualization_results
            }
            
            save_pickle(complete_results, str(reports_dir / "complete_results.pkl"))
            
            # Export JSON summary
            json_results = {
                'summary': {
                    'success': self.results.get('success', False),
                    'total_time': self.results.get('total_pipeline_time', 0.0),
                    'trajectories_generated': self.results.get('total_trajectories', 0),
                    'models_trained': len([k for k, v in model_results.items() if 'error' not in v]),
                    'best_model': self._find_best_model(evaluation_results)
                },
                'detailed_results': complete_results
            }
            
            with open(str(reports_dir / "results_summary.json"), 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            # Generate markdown report
            self._generate_markdown_report(complete_results)
            
            self.logger.info("Results saved to %s", self.output_dir)
            
        except Exception as e:
            self.logger.error("Failed to save results: %s", e)
    
    def _find_best_model(self, evaluation_results: Dict[str, Any]) -> str:
        """Find the best performing model."""
        best_model = "None"
        best_f1 = 0.0
        
        for model_name, results in evaluation_results.items():
            if 'test_metrics' in results:
                f1 = results['test_metrics'].get('f1', 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
        
        return best_model
    
    def _generate_markdown_report(self, complete_results: Dict[str, Any]) -> None:
        """Generate a comprehensive markdown report."""
        # Ensure reports directory exists
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_lines = []
        
        report_lines.append("# AI Agent Trajectory Anomaly Detection - Results Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        pipeline_results = complete_results.get('pipeline_results', {})
        total_time = pipeline_results.get('total_pipeline_time', 0.0)
        total_trajectories = pipeline_results.get('total_trajectories', 0)
        
        report_lines.append(f"- **Total Execution Time:** {total_time:.2f} seconds")
        report_lines.append(f"- **Trajectories Generated:** {total_trajectories}")
        report_lines.append(f"- **Models Trained:** {len(complete_results.get('model_results', {}))}")
        
        best_model = self._find_best_model(complete_results.get('evaluation_results', {}))
        report_lines.append(f"- **Best Performing Model:** {best_model}")
        report_lines.append("")
        
        # Model Performance
        report_lines.append("## Model Performance")
        report_lines.append("")
        
        evaluation_results = complete_results.get('evaluation_results', {})
        if evaluation_results:
            report_lines.append("| Model | F1 Score | Precision | Recall | AUC-ROC |")
            report_lines.append("|-------|----------|-----------|--------|---------|")
            
            for model_name, results in evaluation_results.items():
                if 'test_metrics' in results:
                    metrics = results['test_metrics']
                    f1 = metrics.get('f1', 0.0)
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                    auc_roc = metrics.get('auc_roc', 0.0)
                    
                    report_lines.append(f"| {model_name} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {auc_roc:.3f} |")
        
        report_lines.append("")
        
        # Data Statistics
        report_lines.append("## Dataset Statistics")
        report_lines.append("")
        
        data_splits = pipeline_results.get('data_splits', {})
        report_lines.append(f"- **Training Set:** {data_splits.get('train_size', 0)} samples (100% normal)")
        report_lines.append(f"- **Validation Set:** {data_splits.get('val_size', 0)} samples ({data_splits.get('val_anomaly_rate', 0.0):.1%} anomalous)")
        report_lines.append(f"- **Test Set:** {data_splits.get('test_size', 0)} samples ({data_splits.get('test_anomaly_rate', 0.0):.1%} anomalous)")
        report_lines.append("")
        
        # Feature Engineering
        feature_extraction = pipeline_results.get('feature_extraction', {})
        report_lines.append("## Feature Engineering")
        report_lines.append("")
        report_lines.append(f"- **Total Features Extracted:** {feature_extraction.get('feature_count', 0)}")
        report_lines.append(f"- **Feature Extraction Time:** {feature_extraction.get('extraction_time', 0.0):.2f} seconds")
        report_lines.append("")
        
        # Visualizations
        viz_results = complete_results.get('visualization_results', {})
        if viz_results:
            report_lines.append("## Generated Visualizations")
            report_lines.append("")
            
            for viz_name, viz_data in viz_results.items():
                if isinstance(viz_data, dict) and 'file' in viz_data:
                    file_path = viz_data['file']
                    if file_path:
                        report_lines.append(f"- **{viz_name.replace('_', ' ').title()}:** `{file_path}`")
        
        report_lines.append("")
        
        # Recommendations
        eval_report = evaluation_results.get('evaluation_report', {})
        recommendations = eval_report.get('recommendations', [])
        
        if recommendations:
            report_lines.append("## Recommendations")
            report_lines.append("")
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Configuration
        report_lines.append("## Configuration Used")
        report_lines.append("")
        report_lines.append("```yaml")
        
        import yaml
        config_yaml = yaml.dump(self.config, default_flow_style=False, indent=2)
        report_lines.append(config_yaml)
        report_lines.append("```")
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(reports_dir / "analysis_report.md", 'w') as f:
            f.write(report_content)
        
        self.logger.info("Generated comprehensive markdown report")


def main():
    """Main entry point for the anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="AI Agent Trajectory Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Run with default config
    python main.py --config custom_config.yaml       # Use custom configuration
    python main.py --output-dir results/experiment1  # Specify output directory
    python main.py --debug                           # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    try:
        # Initialize and run pipeline
        pipeline = AnomalyDetectionPipeline(args.config, args.output_dir)
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        if results.get('success', False):
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Total execution time: {results.get('total_pipeline_time', 0.0):.2f} seconds")
            print(f"Results saved to: {args.output_dir}")
            print(f"Generated {results.get('total_trajectories', 0)} trajectories")
            print(f"Trained {len(results.get('model_results', {}))} models")
            
            # Show best model performance
            evaluation_results = results.get('evaluation_results', {})
            best_f1 = 0.0
            best_model = "None"
            
            for model_name, model_eval in evaluation_results.items():
                if 'test_metrics' in model_eval:
                    f1 = model_eval['test_metrics'].get('f1', 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_name
            
            if best_model != "None":
                print(f"Best model: {best_model} (F1: {best_f1:.3f})")
            
            print("\nKey files generated:")
            print(f"  - Analysis report: {args.output_dir}/reports/analysis_report.md")
            print(f"  - Results summary: {args.output_dir}/reports/results_summary.json")
            print(f"  - Charts: {args.output_dir}/charts/")
            print("="*80)
            
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("PIPELINE FAILED")
            print("="*80)
            print(f"Error: {results.get('error', 'Unknown error')}")
            print(f"Check logs in: {args.output_dir}/logs/")
            print("="*80)
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
