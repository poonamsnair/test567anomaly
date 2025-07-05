"""
Enhanced Machine Learning Models for AI Agent Trajectory Anomaly Detection.

This module implements advanced versions of the original models with:
- Bayesian hyperparameter optimization
- Ensemble methods
- Advanced neural architectures
- Graph augmentation techniques
- Improved loss functions
"""

import logging
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR

# Enhanced imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
    from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
    from sklearn.feature_selection import SelectKBest, RFECV, mutual_info_regression
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import VotingRegressor
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    import joblib
except ImportError as e:
    print(f"Warning: Some packages not available: {e}")

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, GraphConv, TransformerConv, SAGEConv, GINConv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.nn import GlobalAttention
    from torch_geometric.utils import dropout_adj, dropout_node
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from .utils import Timer, ensure_directory

logger = logging.getLogger(__name__)


class EnhancedAnomalyDetectionModels:
    """
    Enhanced anomaly detection models with advanced techniques.
    
    Features:
    - Bayesian hyperparameter optimization
    - Ensemble methods
    - Advanced preprocessing
    - Enhanced neural architectures
    - Graph augmentation
    """
    
    def __init__(self, config: Dict):
        """Initialize enhanced models."""
        self.config = config
        self.models_config = config.get('models', {})
        
        # Initialize models
        self.isolation_forest_ensemble = []
        self.one_class_svm_ensemble = []
        self.gnn_autoencoder = None
        
        # Preprocessing components
        self.scalers = {}
        self.feature_selectors = {}
        self.dimensionality_reducers = {}
        
        # Caching for expensive operations
        self.cache = {}
        
        logger.info("Enhanced AnomalyDetectionModels initialized")
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            embeddings: Optional[Dict[str, np.ndarray]] = None,
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare training data by removing any anomaly-related labels.
        Compatible with original models interface.
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for data preparation")
        
        # Remove all anomaly-related columns
        anomaly_columns = [
            'is_anomalous', 'anomaly_severity', 'anomaly_type', 'anomaly_types',
            'graph_id'
        ]
        
        # Create training features by excluding anomaly labels
        if feature_columns is None:
            feature_columns = [col for col in features_df.columns if col not in anomaly_columns]
        training_features = features_df[feature_columns].copy() if set(feature_columns).issubset(features_df.columns) else features_df.copy()
        
        # Handle categorical variables
        categorical_columns = training_features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if training_features[col].apply(lambda x: isinstance(x, list)).any():
                training_features[col] = training_features[col].apply(lambda x: str(x) if isinstance(x, list) else x)
            dummies = pd.get_dummies(training_features[col], prefix=col)
            training_features = training_features.drop(col, axis=1)
            training_features = pd.concat([training_features, dummies], axis=1)
        
        # Ensure column order matches if feature_columns provided
        if feature_columns is not None:
            for col in feature_columns:
                if col not in training_features.columns:
                    training_features[col] = 0
            training_features = training_features[feature_columns]
        else:
            feature_columns = list(training_features.columns)
        
        # Combine with embeddings if provided
        if embeddings:
            embedding_matrix = np.array(list(embeddings.values()))
            embedding_df = pd.DataFrame(
                embedding_matrix, 
                columns=[f'embedding_{i}' for i in range(embedding_matrix.shape[1])]
            )
            training_features = pd.concat([training_features.reset_index(drop=True), 
                                         embedding_df.reset_index(drop=True)], axis=1)
            feature_columns = list(training_features.columns)
        
        # Convert to numpy array
        X = training_features.values.astype(np.float32)
        feature_names = list(training_features.columns)
        
        # Handle NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, feature_names

    def train_enhanced_isolation_forest(self, X: np.ndarray, hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train enhanced Isolation Forest with Bayesian optimization and ensemble methods.
        """
        logger.info("Training Enhanced Isolation Forest with Bayesian optimization")
        
        iso_config = self.models_config.get('isolation_forest', {})
        
        if hyperparameter_tuning:
            # Bayesian optimization
            space = [
                Integer(100, 1000, name='n_estimators'),
                Real(0.01, 0.2, name='contamination'),
                Real(0.3, 1.0, name='max_features'),
                Real(0.3, 1.0, name='max_samples'),
                Categorical([True, False], name='bootstrap'),
                Categorical([True, False], name='warm_start')
            ]
            
            @use_named_args(space)
            def objective(**params):
                try:
                    # Cross-validation score
                    model = IsolationForest(random_state=42, **params)
                    scores = cross_val_score(
                        model, X, cv=5, 
                        scoring='neg_mean_squared_error', 
                        n_jobs=-1
                    )
                    return -np.mean(scores)
                except Exception as e:
                    logger.warning(f"Bayesian optimization iteration failed: {e}")
                    return 1e6
            
            # Bayesian optimization
            n_calls = iso_config.get('n_iterations', 50)
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
            
            best_params = {
                'n_estimators': result.x[0],
                'contamination': result.x[1],
                'max_features': result.x[2],
                'max_samples': result.x[3],
                'bootstrap': result.x[4],
                'warm_start': result.x[5],
                'random_state': 42
            }
            
        else:
            best_params = {
                'n_estimators': 200,
                'contamination': 0.1,
                'max_features': 0.8,
                'max_samples': 0.8,
                'bootstrap': True,
                'random_state': 42
            }
        
        # Train ensemble
        ensemble_config = iso_config.get('ensemble', {})
        if ensemble_config.get('enabled', True):
            n_models = ensemble_config.get('n_models', 5)
            
            models = []
            for i in range(n_models):
                # Add diversity through parameter variation
                params = best_params.copy()
                params['random_state'] = 42 + i
                params['n_estimators'] = int(params['n_estimators'] * (0.8 + 0.4 * np.random.random()))
                
                model = IsolationForest(**params)
                model.fit(X)
                models.append(model)
            
            self.isolation_forest_ensemble = models
            
            # Calculate ensemble scores
            ensemble_scores = np.zeros(len(X))
            for model in models:
                ensemble_scores += model.decision_function(X)
            ensemble_scores /= len(models)
            
            return {
                'models': models,
                'ensemble_scores': ensemble_scores,
                'best_params': best_params,
                'method': 'enhanced_isolation_forest'
            }
        
        else:
            # Single model
            model = IsolationForest(**best_params)
            model.fit(X)
            scores = model.decision_function(X)
            
            return {
                'model': model,
                'training_scores': scores,
                'best_params': best_params,
                'method': 'enhanced_isolation_forest'
            }
    
    def train_enhanced_one_class_svm(self, X: np.ndarray, hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train enhanced One-Class SVM with multi-kernel ensemble and advanced preprocessing.
        """
        logger.info("Training Enhanced One-Class SVM with multi-kernel ensemble")
        
        svm_config = self.models_config.get('one_class_svm', {})
        
        # Advanced preprocessing
        X_processed = self._apply_advanced_preprocessing(X, svm_config)
        
        # Multi-kernel ensemble
        kernels = svm_config.get('kernels', ['rbf', 'linear', 'poly'])
        kernel_models = {}
        
        for kernel in kernels:
            logger.info(f"Training {kernel} kernel SVM")
            
            if hyperparameter_tuning:
                # Bayesian optimization for each kernel
                if kernel == 'rbf':
                    space = [
                        Real(1e-4, 10.0, name='gamma', prior='log-uniform'),
                        Real(0.01, 0.3, name='nu')
                    ]
                    
                    @use_named_args(space)
                    def objective(**params):
                        try:
                            model = OneClassSVM(kernel=kernel, **params)
                            scores = cross_val_score(
                                model, X_processed, cv=3, 
                                scoring='neg_mean_squared_error', 
                                n_jobs=-1
                            )
                            return -np.mean(scores)
                        except:
                            return 1e6
                    
                    result = gp_minimize(objective, space, n_calls=30, random_state=42)
                    best_params = {
                        'gamma': result.x[0],
                        'nu': result.x[1],
                        'kernel': kernel
                    }
                    
                elif kernel == 'poly':
                    space = [
                        Integer(2, 5, name='degree'),
                        Real(0.0, 1.0, name='coef0'),
                        Real(0.01, 0.3, name='nu')
                    ]
                    
                    @use_named_args(space)
                    def objective(**params):
                        try:
                            model = OneClassSVM(kernel=kernel, **params)
                            scores = cross_val_score(
                                model, X_processed, cv=3, 
                                scoring='neg_mean_squared_error', 
                                n_jobs=-1
                            )
                            return -np.mean(scores)
                        except:
                            return 1e6
                    
                    result = gp_minimize(objective, space, n_calls=20, random_state=42)
                    best_params = {
                        'degree': result.x[0],
                        'coef0': result.x[1],
                        'nu': result.x[2],
                        'kernel': kernel
                    }
                    
                else:  # linear, sigmoid
                    space = [Real(0.01, 0.3, name='nu')]
                    
                    @use_named_args(space)
                    def objective(**params):
                        try:
                            model = OneClassSVM(kernel=kernel, **params)
                            scores = cross_val_score(
                                model, X_processed, cv=3, 
                                scoring='neg_mean_squared_error', 
                                n_jobs=-1
                            )
                            return -np.mean(scores)
                        except:
                            return 1e6
                    
                    result = gp_minimize(objective, space, n_calls=15, random_state=42)
                    best_params = {
                        'nu': result.x[0],
                        'kernel': kernel
                    }
                
            else:
                # Default parameters
                best_params = {'kernel': kernel, 'nu': 0.1}
                if kernel == 'rbf':
                    best_params['gamma'] = 'scale'
                elif kernel == 'poly':
                    best_params['degree'] = 3
                    best_params['coef0'] = 0.0
            
            # Train model
            model = OneClassSVM(**best_params)
            model.fit(X_processed)
            
            kernel_models[kernel] = {
                'model': model,
                'params': best_params,
                'scores': model.decision_function(X_processed)
            }
        
        # Ensemble combination
        ensemble_config = svm_config.get('ensemble', {})
        if ensemble_config.get('enabled', True):
            combination_method = ensemble_config.get('combination_method', 'weighted_average')
            
            if combination_method == 'stacking':
                # Meta-learner for stacking
                meta_features = np.column_stack([
                    kernel_models[k]['scores'] for k in kernels
                ])
                meta_learner = Ridge(alpha=1.0)
                # Use dummy targets for unsupervised learning
                dummy_targets = np.zeros(len(X_processed))
                meta_learner.fit(meta_features, dummy_targets)
                
                return {
                    'kernel_models': kernel_models,
                    'meta_learner': meta_learner,
                    'preprocessing': self._get_preprocessing_info(svm_config),
                    'method': 'enhanced_one_class_svm_ensemble'
                }
            else:
                # Weighted average
                weights = self._calculate_kernel_weights(kernel_models, X_processed)
                
                return {
                    'kernel_models': kernel_models,
                    'weights': weights,
                    'preprocessing': self._get_preprocessing_info(svm_config),
                    'method': 'enhanced_one_class_svm_ensemble'
                }
        
        else:
            # Single best kernel
            best_kernel = max(kernel_models.keys(), 
                            key=lambda k: np.mean(kernel_models[k]['scores']))
            
            return {
                'model': kernel_models[best_kernel]['model'],
                'best_params': kernel_models[best_kernel]['params'],
                'preprocessing': self._get_preprocessing_info(svm_config),
                'method': 'enhanced_one_class_svm'
            }
    
    def train_enhanced_gnn_autoencoder(self, graphs: List[Any], hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train enhanced GNN autoencoder with advanced architectures and techniques.
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available")
            return {'error': 'PyTorch Geometric not available'}
        
        logger.info("Training Enhanced GNN Autoencoder")
        
        gnn_config = self.models_config.get('gnn_autoencoder', {})
        
        # Convert graphs with augmentation
        torch_graphs = self._convert_and_augment_graphs(graphs, gnn_config)
        
        if not torch_graphs:
            return {'error': 'No valid graphs'}
        
        if hyperparameter_tuning:
            # Bayesian optimization
            space = [
                Categorical([['64', '128'], ['128', '256'], ['256', '512']], name='hidden_dims'),
                Integer(32, 256, name='latent_dim'),
                Real(1e-4, 1e-1, name='learning_rate', prior='log-uniform'),
                Integer(8, 64, name='batch_size'),
                Real(0.1, 0.5, name='dropout'),
                Real(0.0, 0.01, name='weight_decay', prior='log-uniform'),
                Categorical(['GCN', 'GAT', 'TransformerConv'], name='gnn_type'),
                Categorical(['standard', 'variational', 'attention'], name='architecture')
            ]
            
            @use_named_args(space)
            def objective(**params):
                try:
                    # Convert hidden_dims from string list to int list
                    params['hidden_dims'] = [int(x) for x in params['hidden_dims']]
                    
                    model = self._create_enhanced_gnn_model(torch_graphs[0], params)
                    loss = self._train_gnn_with_early_stopping(model, torch_graphs, params)
                    return loss
                except Exception as e:
                    logger.warning(f"GNN training failed: {e}")
                    return 1e6
            
            n_calls = gnn_config.get('n_iterations', 30)
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
            
            best_params = {
                'hidden_dims': [int(x) for x in result.x[0]],
                'latent_dim': result.x[1],
                'learning_rate': result.x[2],
                'batch_size': result.x[3],
                'dropout': result.x[4],
                'weight_decay': result.x[5],
                'gnn_type': result.x[6],
                'architecture': result.x[7]
            }
            
        else:
            best_params = {
                'hidden_dims': [128, 256],
                'latent_dim': 64,
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout': 0.2,
                'weight_decay': 0.001,
                'gnn_type': 'GAT',
                'architecture': 'variational'
            }
        
        # Train final model
        model = self._create_enhanced_gnn_model(torch_graphs[0], best_params)
        training_scores = self._train_gnn_with_early_stopping(model, torch_graphs, best_params)
        
        return {
            'model': model,
            'best_params': best_params,
            'training_scores': training_scores,
            'method': 'enhanced_gnn_autoencoder'
        }
    
    def _apply_advanced_preprocessing(self, X: np.ndarray, svm_config: Dict) -> np.ndarray:
        """Apply advanced preprocessing for SVM."""
        preprocessing_config = svm_config.get('preprocessing', {})
        
        X_processed = X.copy()
        
        # Scaling
        scaling_methods = preprocessing_config.get('scaling_methods', ['standard'])
        if 'robust' in scaling_methods:
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers['robust'] = scaler
        elif 'quantile' in scaling_methods:
            scaler = QuantileTransformer()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers['quantile'] = scaler
        else:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers['standard'] = scaler
        
        # Dimensionality reduction
        dim_reduction_config = preprocessing_config.get('dimensionality_reduction', {})
        if dim_reduction_config.get('enabled', True):
            methods = dim_reduction_config.get('methods', ['pca'])
            
            if 'kernel_pca' in methods:
                n_components = int(X_processed.shape[1] * 0.9)
                reducer = KernelPCA(n_components=n_components, kernel='rbf')
                X_processed = reducer.fit_transform(X_processed)
                self.dimensionality_reducers['kernel_pca'] = reducer
            else:
                n_components = int(X_processed.shape[1] * 0.9)
                reducer = PCA(n_components=n_components)
                X_processed = reducer.fit_transform(X_processed)
                self.dimensionality_reducers['pca'] = reducer
        
        return X_processed
    
    def _convert_and_augment_graphs(self, graphs: List[Any], gnn_config: Dict) -> List[Data]:
        """Convert graphs to PyTorch Geometric format with augmentation."""
        torch_graphs = []
        
        for graph in graphs:
            if len(graph.nodes) == 0:
                continue
            
            # Convert to PyTorch Geometric
            data = self._convert_single_graph(graph)
            if data is not None:
                torch_graphs.append(data)
        
        # Apply augmentation
        augmentation_config = gnn_config.get('augmentation', {})
        if augmentation_config.get('enabled', True):
            augmented_graphs = []
            
            for data in torch_graphs:
                augmented_graphs.append(data)  # Original
                
                # Node dropout
                if 'node_dropout' in augmentation_config.get('methods', []):
                    dropout_rate = augmentation_config.get('node_dropout_rate', [0.1])[0]
                    augmented_data = self._apply_node_dropout(data, dropout_rate)
                    augmented_graphs.append(augmented_data)
                
                # Edge dropout
                if 'edge_dropout' in augmentation_config.get('methods', []):
                    dropout_rate = augmentation_config.get('edge_dropout_rate', [0.1])[0]
                    augmented_data = self._apply_edge_dropout(data, dropout_rate)
                    augmented_graphs.append(augmented_data)
                
                # Feature masking
                if 'feature_masking' in augmentation_config.get('methods', []):
                    mask_rate = augmentation_config.get('feature_masking_rate', [0.1])[0]
                    augmented_data = self._apply_feature_masking(data, mask_rate)
                    augmented_graphs.append(augmented_data)
            
            return augmented_graphs
        
        return torch_graphs
    
    def _convert_single_graph(self, graph) -> Optional[Data]:
        """Convert a single NetworkX graph to PyTorch Geometric Data."""
        try:
            # Node features
            node_features = []
            node_mapping = {node: i for i, node in enumerate(graph.nodes())}
            
            for node in graph.nodes():
                node_data = graph.nodes[node]
                features = [
                    float(node_data.get('duration', 0.0)),
                    float(node_data.get('performance_score', 0.0)),
                    float(node_data.get('cpu_usage', 0.0)),
                    float(node_data.get('memory_usage', 0.0)),
                    float(node_data.get('network_calls', 0)),
                    float(node_data.get('retry_count', 0)),
                    1.0 if node_data.get('success', True) else 0.0,
                ]
                
                # One-hot encoded node type
                node_type = node_data.get('node_type', 'unknown')
                node_types = ['tool_call', 'reasoning', 'memory_access', 'planning', 
                            'validation', 'handoff', 'observation', 'action']
                
                for nt in node_types:
                    features.append(1.0 if node_type == nt else 0.0)
                
                node_features.append(features)
            
            # Edge features
            edge_indices = []
            edge_attrs = []
            
            for u, v, edge_data in graph.edges(data=True):
                if u in node_mapping and v in node_mapping:
                    edge_indices.append([node_mapping[u], node_mapping[v]])
                    
                    edge_attr = [
                        float(edge_data.get('latency', 0.0)),
                        float(edge_data.get('probability', 1.0)),
                        float(edge_data.get('confidence', 1.0)),
                        float(edge_data.get('success_rate', 1.0)),
                        float(edge_data.get('weight', 1.0))
                    ]
                    edge_attrs.append(edge_attr)
            
            if not edge_indices:
                # Self-loops for isolated nodes
                edge_indices = [[i, i] for i in range(len(node_features))]
                edge_attrs = [[0.0, 1.0, 1.0, 1.0, 1.0] for _ in range(len(node_features))]
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.warning(f"Failed to convert graph: {e}")
            return None
    
    def _apply_node_dropout(self, data: Data, dropout_rate: float) -> Data:
        """Apply node dropout augmentation."""
        num_nodes = data.x.size(0)
        keep_prob = 1 - dropout_rate
        keep_nodes = torch.rand(num_nodes) < keep_prob
        
        if keep_nodes.sum() < 2:  # Keep at least 2 nodes
            keep_nodes[:2] = True
        
        # Filter nodes
        new_x = data.x[keep_nodes]
        
        # Update edge indices
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(torch.where(keep_nodes)[0].tolist())}
        
        new_edge_indices = []
        new_edge_attrs = []
        
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            if u in node_mapping and v in node_mapping:
                new_edge_indices.append([node_mapping[u], node_mapping[v]])
                new_edge_attrs.append(data.edge_attr[i].tolist())
        
        if not new_edge_indices:
            # Add self-loops
            new_edge_indices = [[i, i] for i in range(len(new_x))]
            new_edge_attrs = [[0.0, 1.0, 1.0, 1.0, 1.0] for _ in range(len(new_x))]
        
        new_edge_index = torch.tensor(new_edge_indices, dtype=torch.long).t().contiguous()
        new_edge_attr = torch.tensor(new_edge_attrs, dtype=torch.float)
        
        return Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    
    def _apply_edge_dropout(self, data: Data, dropout_rate: float) -> Data:
        """Apply edge dropout augmentation."""
        edge_index, edge_attr = dropout_adj(
            data.edge_index, 
            data.edge_attr, 
            p=dropout_rate,
            force_undirected=False,
            training=True
        )
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _apply_feature_masking(self, data: Data, mask_rate: float) -> Data:
        """Apply feature masking augmentation."""
        x = data.x.clone()
        num_features = x.size(1)
        num_mask = int(num_features * mask_rate)
        
        if num_mask > 0:
            mask_indices = torch.randperm(num_features)[:num_mask]
            x[:, mask_indices] = 0.0
        
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    
    def _create_enhanced_gnn_model(self, sample_data: Data, params: Dict) -> nn.Module:
        """Create enhanced GNN model based on architecture type."""
        input_dim = sample_data.x.shape[1]
        architecture = params.get('architecture', 'standard')
        
        if architecture == 'variational':
            return EnhancedVariationalGNNAutoencoder(
                input_dim=input_dim,
                hidden_dims=params['hidden_dims'],
                latent_dim=params['latent_dim'],
                gnn_type=params['gnn_type'],
                dropout=params['dropout']
            )
        elif architecture == 'attention':
            return EnhancedAttentionGNNAutoencoder(
                input_dim=input_dim,
                hidden_dims=params['hidden_dims'],
                gnn_type=params['gnn_type'],
                dropout=params['dropout']
            )
        else:
            return EnhancedGNNAutoencoder(
                input_dim=input_dim,
                hidden_dims=params['hidden_dims'],
                gnn_type=params['gnn_type'],
                dropout=params['dropout']
            )
    
    def _train_gnn_with_early_stopping(self, model: nn.Module, torch_graphs: List[Data], params: Dict) -> float:
        """Train GNN with early stopping and learning rate scheduling."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create data loader
        loader = DataLoader(torch_graphs, batch_size=params['batch_size'], shuffle=True)
        
        # Optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 0.0)
        )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Early stopping
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        # Training loop
        model.train()
        for epoch in range(200):  # Max epochs
            epoch_loss = 0.0
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, EnhancedVariationalGNNAutoencoder):
                    reconstructed, mu, logvar = model(batch)
                    loss = model.loss_function(reconstructed, batch.x, mu, logvar)
                else:
                    reconstructed = model(batch)
                    loss = nn.MSELoss()(reconstructed, batch.x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return best_loss
    
    def _calculate_kernel_weights(self, kernel_models: Dict, X: np.ndarray) -> Dict[str, float]:
        """Calculate weights for kernel combination."""
        weights = {}
        total_score = 0.0
        
        for kernel, model_info in kernel_models.items():
            # Use negative mean score as weight (higher is better)
            score = -np.mean(model_info['scores'])
            weights[kernel] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for kernel in weights:
                weights[kernel] /= total_score
        else:
            # Equal weights if all scores are negative
            for kernel in weights:
                weights[kernel] = 1.0 / len(weights)
        
        return weights
    
    def _get_preprocessing_info(self, svm_config: Dict) -> Dict:
        """Get preprocessing information for later use."""
        return {
            'scalers': self.scalers,
            'dimensionality_reducers': self.dimensionality_reducers,
            'config': svm_config.get('preprocessing', {})
        }


class EnhancedGNNAutoencoder(nn.Module):
    """Enhanced GNN Autoencoder with advanced features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], gnn_type: str = 'GAT', dropout: float = 0.2):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            elif gnn_type == 'SAGEConv':
                layer = SAGEConv(prev_dim, hidden_dim)
            else:
                layer = GCNConv(prev_dim, hidden_dim)
            
            self.encoder_layers.append(layer)
            prev_dim = hidden_dim
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        
        for decoder_dim in decoder_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            elif gnn_type == 'SAGEConv':
                layer = SAGEConv(prev_dim, decoder_dim)
            else:
                layer = GCNConv(prev_dim, decoder_dim)
            
            self.decoder_layers.append(layer)
            prev_dim = decoder_dim
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index)
            if i < len(self.decoder_layers) - 1:
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        return x


class EnhancedVariationalGNNAutoencoder(nn.Module):
    """Variational GNN Autoencoder with KL divergence loss."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, 
                 gnn_type: str = 'GAT', dropout: float = 0.2):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.gnn_type = gnn_type
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            else:
                layer = GCNConv(prev_dim, hidden_dim)
            
            self.encoder_layers.append(layer)
            prev_dim = hidden_dim
        
        # Variational layers
        self.mu_layer = GCNConv(prev_dim, latent_dim)
        self.logvar_layer = GCNConv(prev_dim, latent_dim)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_dims = list(reversed(hidden_dims)) + [input_dim]
        prev_dim = latent_dim
        
        for decoder_dim in decoder_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            else:
                layer = GCNConv(prev_dim, decoder_dim)
            
            self.decoder_layers.append(layer)
            prev_dim = decoder_dim
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def encode(self, x, edge_index):
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        mu = self.mu_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, edge_index):
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            z = layer(z, edge_index)
            if i < len(self.decoder_layers) - 1:
                z = torch.relu(z)
                z = self.dropout_layer(z)
        
        return z
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, edge_index)
        
        return reconstructed, mu, logvar
    
    def loss_function(self, reconstructed, original, mu, logvar, beta=1.0):
        """Combined reconstruction and KL divergence loss."""
        reconstruction_loss = nn.MSELoss()(reconstructed, original)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return reconstruction_loss + beta * kl_loss


class EnhancedAttentionGNNAutoencoder(nn.Module):
    """GNN Autoencoder with global attention mechanism."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], gnn_type: str = 'GAT', dropout: float = 0.2):
        super().__init__()
        
        self.gnn_type = gnn_type
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, hidden_dim, heads=8, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, hidden_dim, heads=8, concat=False, dropout=dropout)
            else:
                layer = GCNConv(prev_dim, hidden_dim)
            
            self.encoder_layers.append(layer)
            prev_dim = hidden_dim
        
        # Global attention
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, 1)
            )
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        
        for decoder_dim in decoder_dims:
            if gnn_type == 'GAT':
                layer = GATConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            elif gnn_type == 'TransformerConv':
                layer = TransformerConv(prev_dim, decoder_dim, heads=1, concat=False, dropout=dropout)
            else:
                layer = GCNConv(prev_dim, decoder_dim)
            
            self.decoder_layers.append(layer)
            prev_dim = decoder_dim
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        # Global attention pooling
        if batch is not None:
            x_pooled = self.global_attention(x, batch)
            # Broadcast back to node level
            x = x_pooled[batch] if batch is not None else x_pooled.expand_as(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index)
            if i < len(self.decoder_layers) - 1:
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        return x