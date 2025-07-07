"""
Machine learning models for AI Agent Trajectory Anomaly Detection.

This module implements unsupervised anomaly detection models:
- Isolation Forest: Ensemble-based anomaly detection
- One-Class SVM: Support vector approach for novelty detection
- GNN Diffusion Autoencoder: Deep graph neural network with reconstruction loss

Note: This is pure unsupervised learning - no anomaly labels are used during training.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

# Core ML imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool

from .utils import Timer, ensure_directory

logger = logging.getLogger(__name__)


def _get_dataset_size_params(n_samples: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get adaptive parameters based on dataset size by reading from the config.

    Args:
        n_samples: Number of samples in the dataset.
        config: The main configuration dictionary which should contain 'hyperparameter_grids'.

    Returns:
        Parameter grid for the determined dataset size.
    """
    hyperparam_grids = config.get('hyperparameter_grids', {})

    if n_samples < config.get('dataset_size_thresholds', {}).get('small_medium_boundary', 100):
        grid_type = 'small_dataset'
    elif n_samples < config.get('dataset_size_thresholds', {}).get('medium_large_boundary', 300):
        grid_type = 'medium_dataset'
    else:
        grid_type = 'large_dataset'

    selected_grid = hyperparam_grids.get(grid_type)

    if selected_grid is None:
        logger.warning(f"No hyperparameter grid found for '{grid_type}' in config. Falling back to large_dataset defaults if available, else empty.")
        selected_grid = hyperparam_grids.get('large_dataset', {}) # Fallback strategy
        if not selected_grid:
             logger.error(f"Fallback hyperparameter grid 'large_dataset' also not found. Parameter tuning may fail or use hardcoded defaults.")
             # Minimal fallback to prevent outright crashes, though tuning will be ineffective.
             return {
                'isolation_forest': {}, 'one_class_svm': {}, 'gnn_autoencoder': {}, 'max_combinations': 1
            }

    logger.info(f"Using hyperparameter grid for '{grid_type}' ({n_samples} samples)")
    return selected_grid


def _create_model_result(model: Any, best_params: Dict, training_scores: np.ndarray, 
                        method: str, scaler: Optional[Any] = None, 
                        best_score: Optional[float] = None) -> Dict[str, Any]:
    """Create standardized model result dictionary."""
    result = {
        'model': model,
        'best_params': best_params,
        'training_scores': training_scores,
        'method': method,
        'scaler': scaler
    }
    if best_score is not None:
        result['best_score'] = best_score
    return result


def _process_categorical_features(df: pd.DataFrame, max_categories_for_dummy: int = 5) -> pd.DataFrame:
    """
    Process categorical features with dummy encoding.
    Args:
        df: DataFrame to process.
        max_categories_for_dummy: Maximum number of categories to keep for dummy encoding
                                   when dataset is small. Other categories become 'other'.
    Returns:
        DataFrame with categorical features processed.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Handle list values
        if df[col].apply(lambda x: isinstance(x, list)).any():
            logger.debug(f"Converting list values in column {col} to string")
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        # Limit categories for small datasets
        unique_values = df[col].nunique()
        # The threshold for "small dataset" (len(df) < 100) is hardcoded here but could also be configurable.
        if len(df) < 100 and unique_values > max_categories_for_dummy:
            logger.debug(f"Limiting dummy encoding for column {col} ({unique_values} unique values, "
                         f"max_categories_for_dummy: {max_categories_for_dummy})")
            top_categories = df[col].value_counts().head(max_categories_for_dummy).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
        
        # Convert to dummy variables
        dummies = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1)
        df = pd.concat([df, dummies], axis=1)
    
    return df


def _ensure_feature_consistency(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure feature columns are consistent between train/test splits."""
    if feature_columns is not None:
        # Add missing columns as 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        # Ensure column order matches
        df = df[feature_columns]
        return df, feature_columns
    else:
        return df, list(df.columns)


def _adjust_input_dimensions(features: np.ndarray, expected_dim: int) -> np.ndarray:
    """Adjust input dimensions to match model expectations."""
    if features.shape[1] != expected_dim:
        logger.debug(f"Adjusting input dimensions: {features.shape[1]} -> {expected_dim}")
        if features.shape[1] > expected_dim:
            return features[:, :expected_dim]
        else:
            padding = np.zeros((features.shape[0], expected_dim - features.shape[1]))
            return np.hstack([features, padding])
    return features


def _add_noise_if_constant(scores: np.ndarray, noise_scale: float = 1e-6) -> np.ndarray:
    """Add small noise if all scores are constant."""
    if np.all(scores == scores[0]):
        logger.debug("Adding small noise to constant scores")
        return scores + np.random.normal(0, noise_scale, size=scores.shape)
    return scores


class AnomalyDetectionModels:
    """
    Implements multiple unsupervised anomaly detection models.
    
    This class handles training, evaluation, and inference for:
    1. Isolation Forest - Tree-based ensemble method
    2. One-Class SVM - Support vector novelty detection
    3. GNN Diffusion Autoencoder - Graph neural network approach
    
    All models are trained in purely unsupervised manner without anomaly labels.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize anomaly detection models.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_config = config.get('models', {})
        
        # Initialize models
        self.isolation_forest = None
        self.one_class_svm = None
        self.gnn_autoencoder = None
        
        # Feature scaler
        self.scaler = StandardScaler() if StandardScaler else None
        
        # Best hyperparameters
        self.best_params = {}
        
        logger.info("AnomalyDetectionModels initialized")
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            embeddings: Optional[Dict[str, np.ndarray]] = None,
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare training data by removing anomaly-related labels and processing features.
        
        Args:
            features_df: Feature DataFrame
            embeddings: Optional graph embeddings
            feature_columns: List of feature columns to enforce (from train set)
        
        Returns:
            Tuple of (training_data, feature_names)
        """
        # Remove anomaly-related columns
        anomaly_columns = ['is_anomalous', 'anomaly_severity', 'anomaly_type', 'anomaly_types', 'graph_id']
        
        if feature_columns is None:
            feature_columns = [col for col in features_df.columns if col not in anomaly_columns]
        
        training_features = features_df[feature_columns].copy() if set(feature_columns).issubset(features_df.columns) else features_df.copy()
        
        # Process categorical features
        # Get max_categories from model_config or use a default
        max_cat_dummy = self.models_config.get('max_categories_for_dummy_encoding', 5)
        training_features = _process_categorical_features(training_features, max_cat_dummy)
        
        # Ensure feature consistency
        training_features, feature_columns = _ensure_feature_consistency(training_features, feature_columns)
        
        # Add embeddings if provided
        if embeddings:
            embedding_matrix = np.array(list(embeddings.values()))
            embedding_df = pd.DataFrame(
                embedding_matrix, 
                columns=[f'embedding_{i}' for i in range(embedding_matrix.shape[1])]
            )
            training_features = pd.concat([training_features.reset_index(drop=True), 
                                         embedding_df.reset_index(drop=True)], axis=1)
            feature_columns = list(training_features.columns)
        
        # Convert to numpy array and handle NaN values
        X = training_features.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"Prepared training data with shape: {X.shape}")
        return X, feature_columns
    
    def train_isolation_forest(self, X: np.ndarray, hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train Isolation Forest model with hyperparameter tuning.
        
        Args:
            X: Training data (features only, no labels)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with model and training results
        """
        logger.info(f"Training Isolation Forest on data shape: {X.shape}")
        
        # Apply scaling for better performance
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        
        if hyperparameter_tuning:
            return self._tune_isolation_forest(X_scaled)
        
        # Get default parameters based on dataset size
        n_samples = X_scaled.shape[0]
        # Pass self.config to access hyperparameter_grids
        size_params = _get_dataset_size_params(n_samples, self.config).get('isolation_forest', {})
        
        default_params = {
            'n_estimators': size_params['n_estimators'][1],  # Use middle value
            'contamination': size_params['contamination'][1],
            'max_features': size_params['max_features'][1],
            'max_samples': size_params['max_samples'][1],
            'bootstrap': size_params['bootstrap'][0],  # Prefer False for small datasets
            'random_state': 42
        }
        
        model = IsolationForest(**default_params)
        model.fit(X_scaled)
        scores = model.decision_function(X_scaled)
        
        return _create_model_result(model, default_params, scores, 'isolation_forest', self.scaler)
    
    def _tune_isolation_forest(self, X: np.ndarray) -> Dict[str, Any]:
        """Tune Isolation Forest hyperparameters."""
        n_samples = X.shape[0]
        logger.info(f"Tuning Isolation Forest for {n_samples} samples")
        
        # Get parameter grid based on dataset size
        # Pass self.config to access hyperparameter_grids
        size_params_config = _get_dataset_size_params(n_samples, self.config)
        param_grid = size_params_config.get('isolation_forest', {})
        max_combinations = size_params_config.get('max_combinations', 10) # Default max_combinations
        
        # Create parameter combinations
        all_combinations = list(ParameterGrid(param_grid))
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info(f"Tuning with {len(combinations)} parameter combinations")
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for params in tqdm(combinations, desc="Tuning Isolation Forest"):
            try:
                params['random_state'] = 42
                
                # Validate parameters for small datasets
                if n_samples < 100:
                    params['max_samples'] = min(params['max_samples'], 0.9)
                    params['max_features'] = max(params['max_features'], 0.5)
                
                model = IsolationForest(**params)
                model.fit(X)
                
                scores = model.decision_function(X)
                predictions = model.predict(X)
                
                # Calculate score using silhouette and variance
                if len(set(predictions)) > 1:
                    try:
                        sil_score = silhouette_score(X, predictions)
                    except Exception:
                        sil_score = 0.0
                    
                    score_variance = np.var(scores)
                    
                    # Adaptive scoring heuristic:
                    # The primary goal is to achieve good cluster separation (high silhouette score)
                    # for the predicted inliers vs. outliers.
                    # A secondary goal is to ensure the anomaly scores (`scores`) themselves have some variance;
                    # too little variance might indicate a degenerate model.
                    # However, excessively high variance in scores, especially for small datasets,
                    # might indicate overfitting to a few points, leading to extreme scores.
                    # The heuristic moderately rewards some variance but penalizes very high variance,
                    # especially on smaller datasets, to encourage more stable scoring.
                    # The weights and caps are empirically derived.
                    if n_samples < 100: # Small dataset
                        # Higher weight on silhouette, moderate reward for variance, stronger penalty for high variance
                        score = sil_score * 0.4 + min(score_variance * 3, 0.3) - max(score_variance * 0.1, 0.15)
                    elif n_samples < 300: # Medium dataset
                        # Balanced approach
                        score = sil_score * 0.5 + min(score_variance * 5, 0.4) - max(score_variance * 0.05, 0.1)
                    else: # Larger dataset
                        # Higher reliance on silhouette, gentler reward/penalty for variance
                        score = sil_score * 0.7 + min(score_variance * 10, 0.3) - max(score_variance * 0.02, 0.05)
                else:
                    # If only one cluster is predicted (e.g., all normal or all anomalous), assign a low score.
                    score = -1.0
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                logger.debug(f"Failed to train with params {params}: {e}")
                continue
        
        logger.info(f"Best score: {best_score:.4f}")
        training_scores = best_model.decision_function(X) if best_model else np.zeros(len(X))
        
        return _create_model_result(best_model, best_params, training_scores, 'isolation_forest', self.scaler, best_score)
    
    def train_one_class_svm(self, X: np.ndarray, hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train One-Class SVM model with hyperparameter tuning.
        
        Args:
            X: Training data (features only, no labels)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with model and training results
        """
        logger.info(f"Training One-Class SVM on data shape: {X.shape}")
        
        # Scale features for SVM
        X_scaled = self.scaler.fit_transform(X)
        
        if hyperparameter_tuning:
            return self._tune_one_class_svm(X_scaled)
        
        # Get default parameters based on dataset size
        n_samples = X_scaled.shape[0]
        # Pass self.config to access hyperparameter_grids
        size_params = _get_dataset_size_params(n_samples, self.config).get('one_class_svm', {})
        
        default_params = {
            'kernel': size_params['kernel'][0],  # Use first value
            'gamma': size_params['gamma'][0],
            'nu': size_params['nu'][1]  # Use middle value
        }
        
        model = OneClassSVM(**default_params)
        model.fit(X_scaled)
        scores = model.decision_function(X_scaled)
        
        return _create_model_result(model, default_params, scores, 'one_class_svm', self.scaler)
    
    def _tune_one_class_svm(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Tune One-Class SVM hyperparameters."""
        n_samples = X_scaled.shape[0]
        logger.info(f"Tuning One-Class SVM for {n_samples} samples")
        
        # Get parameter grid
        # Pass self.config to access hyperparameter_grids
        size_params_config = _get_dataset_size_params(n_samples, self.config)
        param_grid = size_params_config.get('one_class_svm', {})
        # Note: max_combinations for OCSVM tuning was hardcoded to 10.
        # We'll use the one from config or a default if not specified for consistency.
        max_combinations = size_params_config.get('max_combinations_ocsvm',
                                               size_params_config.get('max_combinations', 10))

        # Create parameter combinations
        all_combinations = list(ParameterGrid(param_grid))
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info(f"Tuning One-Class SVM with {len(combinations)} parameter combinations (max: {max_combinations})")
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        for params in tqdm(combinations, desc="Tuning One-Class SVM"):
            try:
                model = OneClassSVM(**params)
                model.fit(X_scaled)
                
                scores = model.decision_function(X_scaled)
                predictions = model.predict(X_scaled)
                
                # Calculate score using silhouette and variance
                if len(set(predictions)) > 1:
                    try:
                        sil_score = silhouette_score(X_scaled, predictions)
                    except Exception:
                        sil_score = 0.0
                    
                    score_variance = np.var(scores)
                    
                    # Adaptive scoring heuristic for One-Class SVM:
                    # Similar to Isolation Forest, prioritize good cluster separation (silhouette score).
                    # For OCSVM, which defines a boundary around normal data, a very tight boundary might lead
                    # to high silhouette if it correctly separates some noise, but it might also be overfit.
                    # A very loose boundary might result in poor separation.
                    # This heuristic penalizes high score variance more directly than in Isolation Forest,
                    # as OCSVM decision_function scores can sometimes have extreme values for points far
                    # from the boundary, potentially indicating overfitting if the boundary is too complex.
                    # The aim is to find a balance between good separation and a stable, generalizable boundary.
                    # The weights and caps are empirically derived.
                    if n_samples < 100: # Small dataset
                        # Emphasize silhouette, but with a noticeable penalty for high variance
                        score = sil_score * 0.6 - max(score_variance * 0.2, 0.15)
                    elif n_samples < 300: # Medium dataset
                        # Similar emphasis, slightly milder variance penalty
                        score = sil_score * 0.7 - max(score_variance * 0.1, 0.1)
                    else: # Larger dataset
                        # Primarily rely on silhouette score, with a smaller variance penalty
                        score = sil_score * 0.8 - max(score_variance * 0.05, 0.05)
                else:
                    # If only one cluster is predicted, assign a low score.
                    score = -1.0
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                logger.debug(f"Failed to train with params {params}: {e}")
                continue
        
        logger.info(f"Best score: {best_score:.4f}")
        training_scores = best_model.decision_function(X_scaled) if best_model else np.zeros(len(X_scaled))
        
        return _create_model_result(best_model, best_params, training_scores, 'one_class_svm', self.scaler, best_score)
    
    def train_gnn_autoencoder(self, graphs: List[Any], hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train GNN Diffusion Autoencoder model.
        
        Args:
            graphs: List of NetworkX graphs
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with model and training results
        """
        logger.info(f"Training GNN Autoencoder on {len(graphs)} graphs")
        
        # Convert NetworkX graphs to PyTorch Geometric format
        torch_graphs = self._convert_to_torch_geometric(graphs)
        
        if not torch_graphs:
            logger.warning("No valid graphs for GNN training")
            return _create_model_result(None, {}, np.array([]), 'gnn_autoencoder', error='No valid graphs')
        
        if hyperparameter_tuning:
            return self._tune_gnn_autoencoder(torch_graphs)
        
        # Get default parameters based on dataset size
        n_graphs = len(torch_graphs)
        # Pass self.config to access hyperparameter_grids
        size_params = _get_dataset_size_params(n_graphs, self.config).get('gnn_autoencoder', {})
        
        default_params = {
            'hidden_dims': size_params['hidden_dims'][0],  # Use first value
            'learning_rate': size_params['learning_rate'][0],
            'dropout': size_params['dropout'][0],
            'batch_size': size_params['batch_size'][0],
            'epochs': size_params['epochs'][0],
            'gnn_type': size_params['gnn_type'][0]
        }
        
        model = self._create_gnn_model(torch_graphs[0], default_params)
        training_scores = self._train_gnn_model(model, torch_graphs, default_params)
        
        return _create_model_result(model, default_params, training_scores, 'gnn_autoencoder')
    
    def _convert_to_torch_geometric(self, graphs: List[Any]) -> List[Data]:
        """Convert NetworkX graphs to PyTorch Geometric Data objects with small graph optimizations and consistent node feature dimensions."""
        torch_graphs = []
        expected_dim = None
        for graph in graphs:
            try:
                if len(graph.nodes) == 0:
                    continue
                if len(graph.nodes) < 2:
                    logger.debug("Skipping graph with only %d nodes", len(graph.nodes))
                    continue
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
                    node_type = node_data.get('node_type', 'unknown')
                    node_types = ['tool_call', 'reasoning', 'memory_access', 'planning', 
                                'validation', 'handoff', 'observation', 'action']
                    for nt in node_types:
                        features.append(1.0 if node_type == nt else 0.0)
                    node_features.append(features)
                # Ensure consistent feature dimension
                if expected_dim is None:
                    expected_dim = len(node_features[0])
                for i in range(len(node_features)):
                    feat = node_features[i]
                    if len(feat) < expected_dim:
                        node_features[i] = feat + [0.0] * (expected_dim - len(feat))
                    elif len(feat) > expected_dim:
                        node_features[i] = feat[:expected_dim]
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
                    edge_indices = [[i, i] for i in range(len(node_features))]
                    edge_attrs = [[0.0, 1.0, 1.0, 1.0, 1.0] for _ in range(len(node_features))]
                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                torch_graphs.append(data)
            except Exception as e:
                logger.warning("Failed to convert graph: %s", e)
                continue
        logger.info("Converted %d graphs to PyTorch Geometric format", len(torch_graphs))
        return torch_graphs
    
    def _create_gnn_model(self, sample_data: Data, params: Dict) -> nn.Module:
        """Create GNN autoencoder model with consistent input dimension."""
        # Handle the case where x might not be available
        if hasattr(sample_data, 'x') and sample_data.x is not None:
            input_dim = sample_data.x.shape[1]
        else:
            # Default input dimension if x is not available
            input_dim = 15  # Based on the feature extraction above
        
        # Store input dimension in params for consistency
        params['input_dim'] = input_dim
        
        hidden_dims = params['hidden_dims']
        gnn_type = params['gnn_type']
        dropout = params['dropout']
        
        # Adjust hidden dimensions if they're too large compared to input
        if input_dim < 20:
            adjusted_hidden_dims = []
            for dim in hidden_dims:
                if dim > input_dim * 2:
                    adjusted_dim = max(input_dim, dim // 2)
                    logger.warning("Hidden dimension %d is very large compared to input %d, reducing to %d", 
                                  dim, input_dim, adjusted_dim)
                    adjusted_hidden_dims.append(adjusted_dim)
                else:
                    adjusted_hidden_dims.append(dim)
            hidden_dims = adjusted_hidden_dims
        
        return GNNAutoencoder(input_dim, hidden_dims, gnn_type, dropout)
    
    def _train_gnn_model(self, model: nn.Module, torch_graphs: List[Data], params: Dict) -> np.ndarray:
        """Train the GNN autoencoder model with small dataset optimizations."""
        if torch is None:
            return np.array([])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Adaptive batch size for small datasets
        n_graphs = len(torch_graphs)
        batch_size = params['batch_size']
        
        # Ensure batch size doesn't exceed dataset size
        if batch_size > n_graphs:
            batch_size = max(1, n_graphs // 2)  # Use at least 2 batches
            logger.info("Adjusted batch size from %d to %d for small dataset (%d graphs)", 
                       params['batch_size'], batch_size, n_graphs)
        
        # Create data loader
        loader = DataLoader(torch_graphs, batch_size=batch_size, shuffle=True)
        
        # Optimizer with adaptive learning rate
        learning_rate = params['learning_rate']
        if n_graphs < 50:
            # Lower learning rate for very small datasets
            learning_rate *= 0.5
            logger.info("Reduced learning rate to %.6f for small dataset", learning_rate)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping for small datasets
        patience = 10 if n_graphs < 100 else 20
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        model.train()
        training_losses = []
        
        for epoch in range(params['epochs']):
            epoch_loss = 0.0
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = model(batch)
                
                # Reconstruction loss
                loss = nn.MSELoss()(reconstructed, batch.x)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if n_graphs < 100:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            training_losses.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience: %d)", epoch + 1, patience)
                break
            
            if epoch % 20 == 0:
                logger.info("Epoch %d/%d, Loss: %.6f", epoch, params['epochs'], training_losses[-1])
        
        # Calculate reconstruction errors for training data
        model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                
                # Calculate per-graph reconstruction error
                if hasattr(batch, 'num_graphs') and batch.num_graphs > 1:
                    for i in range(batch.num_graphs):
                        graph_mask = batch.batch == i
                        original = batch.x[graph_mask]
                        recon = reconstructed[graph_mask]
                        
                        error = torch.mean((original - recon) ** 2).item()
                        reconstruction_errors.append(error)
                else:
                    # Single graph case
                    error = torch.mean((batch.x - reconstructed) ** 2).item()
                    reconstruction_errors.append(error)
        
        return np.array(reconstruction_errors)
    
    def _tune_gnn_autoencoder(self, torch_graphs: List[Data]) -> Dict[str, Any]:
        """Tune GNN autoencoder hyperparameters with small sample optimizations."""
        n_graphs = len(torch_graphs)
        logger.info("Tuning GNN Autoencoder for %d graphs", n_graphs)

        # Get the appropriate grid from config based on dataset size
        # This returns the entire grid for that size (IF, OCSVM, GNN params, max_combinations)
        size_specific_config = _get_dataset_size_params(n_graphs, self.config)

        # Extract GNN specific part and max_combinations
        param_grid_gnn = size_specific_config.get('gnn_autoencoder', {})
        if not param_grid_gnn:
            logger.warning("GNN autoencoder params not found in the selected size_specific_config. Using empty grid.")
            param_grid_gnn = {} # Ensure it's a dict

        # Max combinations for GNN tuning can be specific or general
        max_combinations = size_specific_config.get('max_combinations_gnn',
                                                 size_specific_config.get('max_combinations', 5))

        all_combinations = list(ParameterGrid(param_grid_gnn)) if ParameterGrid and param_grid_gnn else []
        
        if not all_combinations:
            logger.warning("No GNN parameter combinations to tune. Skipping GNN tuning.")
            return {
                'model': None, 'best_params': None, 'best_score': np.inf,
                'training_scores': np.array([]), 'method': 'gnn_autoencoder'
            }

        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)  # For reproducibility
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info("Tuning GNN Autoencoder with %d parameter combinations", len(combinations))
        
        best_score = np.inf  # Lower reconstruction error is better
        best_params = None
        best_model = None
        best_training_scores = None
        
        progress_bar = tqdm(combinations, desc="Tuning GNN Autoencoder") if tqdm else combinations
        
        for params in progress_bar:
            try:
                model = self._create_gnn_model(torch_graphs[0], params)
                training_scores = self._train_gnn_model(model, torch_graphs, params)
                
                # Use mean reconstruction error as score
                score = np.mean(training_scores)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    best_training_scores = training_scores
                    
            except Exception as e:
                logger.warning("Failed to train GNN with params %s: %s", params, e)
                continue
        
        logger.info("Best GNN Autoencoder score: %.6f with params: %s", best_score, best_params)
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'training_scores': best_training_scores if best_training_scores is not None else np.array([]),
            'method': 'gnn_autoencoder'
        }
    
    def predict_anomalies_graphs(self, model_results: Dict[str, Any], test_graphs: List[Any], threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies for a list of test graphs using the trained GNN autoencoder.
        Returns per-graph reconstruction errors as anomaly scores.
        """
        model = model_results.get('model')
        method = model_results.get('method')
        if model is None or method != 'gnn_autoencoder' or torch is None:
            logger.warning("No trained GNN model available for graph prediction")
            return np.zeros(len(test_graphs)), np.zeros(len(test_graphs))
        try:
            torch_graphs = self._convert_to_torch_geometric(test_graphs)
            device = next(model.parameters()).device
            model.eval()
            scores = []
            with torch.no_grad():
                for data in torch_graphs:
                    data = data.to(device)
                    recon = model(data)
                    error = torch.mean((data.x - recon) ** 2).item()
                    scores.append(error)
            scores = np.array(scores)
            if threshold is None:
                threshold = np.percentile(scores, 90)
            predictions = (scores > threshold).astype(int)
            return scores, predictions
        except Exception as e:
            logger.error("Error in GNN graph prediction: %s", e)
            return np.zeros(len(test_graphs)), np.zeros(len(test_graphs))

    def predict_anomalies(self, model_results: Dict[str, Any], X_test: Union[np.ndarray, List[Any]], 
                         threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using trained model. Supports both flat features and graph input for GNN.
        """
        model = model_results.get('model')
        method = model_results.get('method')
        
        # Handle ensemble models
        if method == 'ensemble' or 'ensemble_model' in model_results:
            logger.info("Processing ensemble model prediction")
            ensemble_model = model_results.get('ensemble_model')
            if ensemble_model is None:
                logger.warning("No ensemble model found in model_results")
                return np.zeros(len(X_test)), np.zeros(len(X_test))
            
            try:
                scores, predictions = ensemble_model.predict_anomalies(X_test, threshold)
                logger.info(f"Ensemble prediction completed - {np.sum(predictions)} anomalies detected")
                return scores, predictions
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                return np.zeros(len(X_test)), np.zeros(len(X_test))
        
        if model is None:
            logger.warning("No trained model available for prediction")
            return np.zeros(len(X_test)), np.zeros(len(X_test))
        
        try:
            if method == 'isolation_forest':
                X_test_scaled = model_results.get('scaler').transform(X_test) if model_results.get('scaler') else X_test
                # Standardize scores: negate to make higher scores more anomalous
                scores = -model.decision_function(X_test_scaled)
                # Default threshold percentile might need adjustment based on negated scores,
                # e.g., percentile of 15 for original scores becomes 85 for negated scores.
                # For now, we assume threshold is externally calibrated or this default is acceptable.
                if threshold is None:
                    # If lower scores were anomalous (e.g., 15th percentile),
                    # then for negated scores, higher scores are anomalous (85th percentile).
                    threshold = np.percentile(scores, 85) # Keep consistent with "higher is anomalous"
                
            elif method == 'one_class_svm':
                X_test_scaled = model_results.get('scaler').transform(X_test) if model_results.get('scaler') else X_test
                # Standardize scores: negate to make higher scores more anomalous
                scores = -model.decision_function(X_test_scaled)
                if threshold is None:
                    # If lower scores were anomalous (e.g., 5th percentile for OCSVM),
                    # then for negated scores, higher scores are anomalous (95th percentile).
                    threshold = np.percentile(scores, 95) # Keep consistent with "higher is anomalous"

            elif method == 'gnn_autoencoder':
                if isinstance(X_test, list):
                    return self.predict_anomalies_graphs(model_results, X_test, threshold)
                
                # Fallback for flat features
                if hasattr(model, 'forward'):
                    try:
                        device = next(model.parameters()).device
                        expected_input_dim = model.encoder_layers[0].in_channels if hasattr(model, 'encoder_layers') and model.encoder_layers else X_test.shape[1]
                        
                        X_test_adjusted = _adjust_input_dimensions(X_test, expected_input_dim)
                        
                        dummy_x = torch.tensor(X_test_adjusted, dtype=torch.float).to(device)
                        dummy_edge_index = torch.tensor([[i, i] for i in range(len(X_test_adjusted))], dtype=torch.long).t().contiguous().to(device)
                        dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)
                        
                        with torch.no_grad():
                            reconstructed = model(dummy_data)
                            scores = torch.mean((dummy_x - reconstructed) ** 2, dim=1).cpu().numpy()
                    except Exception as e:
                        logger.warning(f"GNN prediction failed: {e}, using random scores")
                        scores = np.random.random(len(X_test))
                else:
                    scores = np.random.random(len(X_test))
                
                if threshold is None:
                    threshold = np.percentile(scores, 90) # Higher score is anomalous
                # Universal prediction logic after score standardization
            else:
                logger.warning(f"Unknown model method: {method}")
                scores = np.zeros(len(X_test))
                # predictions will also be all zeros

            # Universal prediction logic: higher scores are anomalous
            predictions = (scores > threshold).astype(int)
            
            # Add noise if scores are constant
            scores = _add_noise_if_constant(scores)
            
            return scores, predictions
            
        except Exception as e:
            logger.error(f"Error in anomaly prediction: {e}")
            return np.zeros(len(X_test)), np.zeros(len(X_test))
    
    def train_ensemble_model(self, base_models: Dict[str, Any], val_features: np.ndarray, 
                           val_labels: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble model using pre-trained base models.
        
        Args:
            base_models: Dictionary of trained base models
            val_features: Validation features for weight optimization
            val_labels: Validation labels (1 for anomalous, 0 for normal)
        
        Returns:
            Dictionary containing ensemble model results
        """
        logger.info("Training ensemble model with %d base models", len(base_models))
        
        # Check if ensemble is enabled in config
        ensemble_config = self.config.get('ensemble', {})
        if not ensemble_config.get('enabled', True):
            logger.info("Ensemble training disabled in config")
            return {}
        
        # Create ensemble detector
        ensemble_detector = EnsembleAnomalyDetector(self.config, base_models)
        
        # Optimize weights using validation data
        weights = ensemble_detector.optimize_weights(val_features, val_labels)
        
        # Create ensemble model results
        ensemble_results = {
            'ensemble_model': ensemble_detector,
            'method': 'ensemble',
            'weights': weights,
            'base_models': list(base_models.keys()),
            'optimization_method': ensemble_detector.optimization_method,
            'fusion_method': ensemble_detector.fusion_method,
            'confidence_weighting': ensemble_detector.confidence_weighting,
            'training_info': {
                'num_base_models': len(base_models),
                'valid_models': len(weights),
                'weight_sum': sum(weights.values())
            }
        }
        
        logger.info("Ensemble model trained successfully with weights: %s", weights)
        return ensemble_results
    
    def save_models(self, models: Dict[str, Any], filepath: str) -> None:
        """Save trained models to file."""
        from pathlib import Path
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        
        logger.info(f"Saved models to {filepath}")
    
    def load_models(self, filepath: str) -> Dict[str, Any]:
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
        
        logger.info(f"Loaded models from {filepath}")
        return models


class GNNAutoencoder(nn.Module):
    """
    Graph Neural Network Autoencoder for anomaly detection.
    
    This model uses graph convolutions to encode node features into a latent space
    and then reconstructs the original features. Reconstruction error is used
    as an anomaly score.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 gnn_type: str = 'GCN', dropout: float = 0.1):
        """
        Initialize GNN autoencoder with small dataset optimizations.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            gnn_type: Type of GNN layer ('GCN', 'GAT', 'GraphConv')
            dropout: Dropout rate
        """
        super(GNNAutoencoder, self).__init__()
        
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Validate input dimensions for small datasets
        if input_dim < 5:
            logger.warning("Very small input dimension (%d), this may affect GNN performance", input_dim)
        
        # Ensure hidden dimensions are reasonable for small datasets
        validated_hidden_dims = []
        for dim in hidden_dims:
            if dim < 8:
                logger.warning("Hidden dimension %d is very small, increasing to 8", dim)
                validated_hidden_dims.append(8)
            elif dim > input_dim * 4:
                logger.warning("Hidden dimension %d is very large compared to input %d, reducing", dim, input_dim)
                validated_hidden_dims.append(min(dim, input_dim * 2))
            else:
                validated_hidden_dims.append(dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in validated_hidden_dims:
            if gnn_type == 'GCN':
                layer = GCNConv(prev_dim, hidden_dim)
            elif gnn_type == 'GAT':
                layer = GATConv(prev_dim, hidden_dim)
            elif gnn_type == 'GraphConv':
                layer = GraphConv(prev_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.encoder_layers.append(layer)
            prev_dim = hidden_dim
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        
        decoder_dims = list(reversed(validated_hidden_dims[:-1])) + [input_dim]
        for decoder_dim in decoder_dims:
            if gnn_type == 'GCN':
                layer = GCNConv(prev_dim, decoder_dim)
            elif gnn_type == 'GAT':
                layer = GATConv(prev_dim, decoder_dim)
            elif gnn_type == 'GraphConv':
                layer = GraphConv(prev_dim, decoder_dim)
            
            self.decoder_layers.append(layer)
            prev_dim = decoder_dim
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data):
        """Forward pass through the autoencoder with small dataset optimizations."""
        x, edge_index = data.x, data.edge_index
        
        # Handle edge cases for small datasets
        if x.shape[0] == 0:
            logger.warning("Empty input tensor in GNN forward pass")
            return x
        
        # Ensure edge_index is valid
        if edge_index.shape[1] == 0:
            # Create self-loops if no edges
            edge_index = torch.stack([
                torch.arange(x.shape[0], device=x.device),
                torch.arange(x.shape[0], device=x.device)
            ])
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:  # No activation on last encoder layer
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index)
            if i < len(self.decoder_layers) - 1:  # No activation on output layer
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        return x


class EnsembleAnomalyDetector:
    """
    Ensemble model for anomaly detection that combines multiple base models.
    
    This ensemble uses weighted averaging of anomaly scores from different models,
    with weights optimized based on validation performance or other criteria.
    """
    
    def __init__(self, config: Dict, base_models: Dict[str, Any]):
        """
        Initialize ensemble detector.
        
        Args:
            config: Configuration dictionary with ensemble settings
            base_models: Dictionary of trained base models
        """
        self.config = config
        self.ensemble_config = config.get('ensemble', {})
        self.base_models = base_models
        
        # Initialize weights (will be optimized later)
        self.weights = None
        self.optimization_method = self.ensemble_config.get('optimization_method', 'validation_performance')
        self.fusion_method = self.ensemble_config.get('fusion_method', 'weighted_average')
        self.confidence_weighting = self.ensemble_config.get('confidence_weighting', True)
        
        # Weight constraints
        weight_constraints = self.ensemble_config.get('weight_constraints', {})
        self.min_weight = weight_constraints.get('min_weight', 0.0)
        self.max_weight = weight_constraints.get('max_weight', 1.0)
        self.sum_to_one = weight_constraints.get('sum_to_one', True)
        
        # Base model selection
        self.base_model_names = self.ensemble_config.get('base_models', ['isolation_forest', 'one_class_svm', 'gnn_autoencoder'])
        
        logger.info("EnsembleAnomalyDetector initialized with %d base models", len(base_models))
    
    def optimize_weights(self, val_features: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            val_features: Validation features
            val_labels: Validation labels (1 for anomalous, 0 for normal)
        
        Returns:
            Dictionary of optimized weights for each base model
        """
        logger.info("Optimizing ensemble weights using %s method with %d validation samples", 
                   self.optimization_method, len(val_features))
        
        # Get predictions from all base models
        base_scores = {}
        valid_models = []
        
        for model_name in self.base_model_names:
            if model_name in self.base_models:
                try:
                    logger.debug("Getting scores from %s for weight optimization", model_name)
                    model_data = self.base_models[model_name]
                    scores, _ = self._get_base_model_scores(model_data, val_features, model_name)
                    if scores is not None and len(scores) > 0:
                        base_scores[model_name] = scores
                        valid_models.append(model_name)
                        logger.debug("Successfully got scores from %s: range=[%.4f, %.4f]", 
                                   model_name, np.min(scores), np.max(scores))
                    else:
                        logger.warning("No valid scores from %s for weight optimization", model_name)
                except Exception as e:
                    logger.warning("Failed to get scores from %s for weight optimization: %s", model_name, e)
        
        logger.info("Valid models for weight optimization: %s", valid_models)
        
        if len(valid_models) < 2:
            logger.warning("Not enough valid base models for ensemble (%d), using equal weights", len(valid_models))
            equal_weights = {model: 1.0 / len(valid_models) for model in valid_models}
            logger.info("Equal weights assigned: %s", equal_weights)
            return equal_weights
        
        # Optimize weights based on method
        logger.debug("Starting weight optimization with method: %s", self.optimization_method)
        
        if self.optimization_method == 'validation_performance':
            weights = self._optimize_by_performance(base_scores, val_labels, valid_models)
        elif self.optimization_method == 'diversity_maximization':
            weights = self._optimize_by_diversity(base_scores, valid_models)
        elif self.optimization_method == 'confidence_weighted':
            weights = self._optimize_by_confidence(base_scores, val_labels, valid_models)
        else:
            logger.warning("Unknown optimization method: %s, using equal weights", self.optimization_method)
            weights = {model: 1.0 / len(valid_models) for model in valid_models}
        
        logger.debug("Raw weights before constraints: %s", weights)
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights)
        
        self.weights = weights
        logger.info("Optimized weights: %s", weights)
        logger.info("Weight sum: %.4f", sum(weights.values()))
        
        return weights
    
    def _optimize_by_performance(self, base_scores: Dict[str, np.ndarray], 
                                val_labels: np.ndarray, valid_models: List[str]) -> Dict[str, float]:
        """Optimize weights based on individual model performance on validation set."""
        try:
            from sklearn.metrics import roc_auc_score
            
            logger.debug("Starting performance-based weight optimization")
            
            # Calculate ROC AUC for each model
            model_performances = {}
            for model_name in valid_models:
                try:
                    auc_score = roc_auc_score(val_labels, base_scores[model_name])
                    model_performances[model_name] = auc_score
                    logger.debug("Model %s AUC: %.4f", model_name, auc_score)
                except Exception as e:
                    logger.warning("Failed to calculate AUC for %s: %s", model_name, e)
                    model_performances[model_name] = 0.5  # Random performance
            
            logger.debug("Model performances: %s", model_performances)
            
            # Convert AUC scores to weights (higher AUC = higher weight)
            total_performance = sum(model_performances.values())
            if total_performance > 0:
                weights = {model: perf / total_performance for model, perf in model_performances.items()}
                logger.debug("Performance-based weights: %s", weights)
            else:
                weights = {model: 1.0 / len(valid_models) for model in valid_models}
                logger.warning("Zero total performance, using equal weights")
            
            return weights
            
        except Exception as e:
            logger.error("Performance-based optimization failed: %s", e)
            return {model: 1.0 / len(valid_models) for model in valid_models}
    
    def _optimize_by_diversity(self, base_scores: Dict[str, np.ndarray], 
                              valid_models: List[str]) -> Dict[str, float]:
        """Optimize weights to maximize diversity among base models."""
        try:
            # Calculate pairwise correlations between model predictions
            correlations = {}
            for i, model1 in enumerate(valid_models):
                for j, model2 in enumerate(valid_models[i+1:], i+1):
                    corr = np.corrcoef(base_scores[model1], base_scores[model2])[0, 1]
                    correlations[(model1, model2)] = abs(corr)
            
            # Calculate diversity score for each model (lower average correlation = higher diversity)
            diversity_scores = {}
            for model in valid_models:
                model_correlations = [corr for (m1, m2), corr in correlations.items() 
                                    if m1 == model or m2 == model]
                if model_correlations:
                    avg_correlation = np.mean(model_correlations)
                    diversity_scores[model] = 1.0 - avg_correlation  # Higher diversity = higher weight
                else:
                    diversity_scores[model] = 1.0
            
            # Convert diversity scores to weights
            total_diversity = sum(diversity_scores.values())
            if total_diversity > 0:
                weights = {model: div / total_diversity for model, div in diversity_scores.items()}
            else:
                weights = {model: 1.0 / len(valid_models) for model in valid_models}
            
            return weights
            
        except Exception as e:
            logger.error("Diversity-based optimization failed: %s", e)
            return {model: 1.0 / len(valid_models) for model in valid_models}
    
    def _optimize_by_confidence(self, base_scores: Dict[str, np.ndarray], 
                               val_labels: np.ndarray, valid_models: List[str]) -> Dict[str, float]:
        """Optimize weights based on model confidence (agreement with other models)."""
        try:
            # Calculate confidence scores based on agreement with other models
            confidence_scores = {}
            
            for model in valid_models:
                # Calculate how much this model agrees with the majority
                model_scores = base_scores[model]
                other_scores = [base_scores[m] for m in valid_models if m != model]
                
                if other_scores:
                    # Calculate average of other models
                    avg_other_scores = np.mean(other_scores, axis=0)
                    
                    # Calculate correlation with average of other models
                    correlation = np.corrcoef(model_scores, avg_other_scores)[0, 1]
                    confidence_scores[model] = max(0, correlation)  # Ensure non-negative
                else:
                    confidence_scores[model] = 1.0
            
            # Convert confidence scores to weights
            total_confidence = sum(confidence_scores.values())
            if total_confidence > 0:
                weights = {model: conf / total_confidence for model, conf in confidence_scores.items()}
            else:
                weights = {model: 1.0 / len(valid_models) for model in valid_models}
            
            return weights
            
        except Exception as e:
            logger.error("Confidence-based optimization failed: %s", e)
            return {model: 1.0 / len(valid_models) for model in valid_models}
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints (min/max weights, sum to one)."""
        # Apply min/max constraints
        constrained_weights = {}
        for model, weight in weights.items():
            constrained_weight = max(self.min_weight, min(self.max_weight, weight))
            constrained_weights[model] = constrained_weight
        
        # Apply sum-to-one constraint if requested
        if self.sum_to_one:
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {model: weight / total_weight 
                                     for model, weight in constrained_weights.items()}
        
        return constrained_weights
    
    def _get_base_model_scores(self, model_data: Dict[str, Any], 
                              features: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly scores from a base model."""
        try:
            logger.debug("Getting scores from base model: %s", model_name)
            
            # Check if it's an ensemble model (recursive case)
            if model_name == 'ensemble_model' or 'ensemble_model' in model_data:
                logger.warning("Nested ensemble detected, skipping")
                return np.zeros(len(features)), np.zeros(len(features))
            
            # Get the model and method
            model = model_data.get('model')
            method = model_data.get('method')
            
            if model is None:
                logger.warning("No model found for %s", model_name)
                return np.zeros(len(features)), np.zeros(len(features))
            
            # Get scores based on model type
            # ALL SCORES ARE TRANSFORMED SO HIGHER = MORE ANOMALOUS
            if method == 'isolation_forest':
                logger.debug("Processing Isolation Forest scores for %s", model_name)
                raw_scores = model.score_samples(features) # Higher is more normal
                scores = -raw_scores # Negate to make higher more anomalous
                logger.debug("Isolation Forest standardized scores range: [%.4f, %.4f]", np.min(scores), np.max(scores))
            elif method == 'one_class_svm':
                logger.debug("Processing One-Class SVM scores for %s", model_name)
                scaler = model_data.get('scaler')
                pca = model_data.get('pca')
                variance_threshold = model_data.get('variance_threshold')
                if scaler is None:
                    logger.warning("No scaler found for %s", model_name)
                    scores = np.zeros(len(features))
                else:
                    features_processed = features.copy()
                    if variance_threshold is not None:
                        features_processed = variance_threshold.transform(features_processed)
                    if pca is not None:
                        features_processed = pca.transform(features_processed)
                    features_scaled = scaler.transform(features_processed)
                    raw_scores = model.decision_function(features_scaled) # Lower is more anomalous
                    scores = -raw_scores # Negate to make higher more anomalous
                    logger.debug("One-Class SVM standardized scores range: [%.4f, %.4f]", np.min(scores), np.max(scores))
            elif method == 'gnn_autoencoder':
                logger.debug("Processing GNN Autoencoder scores for %s", model_name)
                if hasattr(model, 'forward'):
                    try:
                        device = next(model.parameters()).device
                        expected_input_dim = model.encoder_layers[0].in_channels if hasattr(model, 'encoder_layers') and model.encoder_layers else features.shape[1]
                        if features.shape[1] != expected_input_dim:
                            logger.warning("Input dimension mismatch for GNN: expected %d, got %d. Adjusting input.", expected_input_dim, features.shape[1])
                            if features.shape[1] > expected_input_dim:
                                features_adjusted = features[:, :expected_input_dim]
                            else:
                                padding = np.zeros((features.shape[0], expected_input_dim - features.shape[1]))
                                features_adjusted = np.hstack([features, padding])
                        else:
                            features_adjusted = features
                        dummy_x = torch.tensor(features_adjusted, dtype=torch.float).to(device)
                        dummy_edge_index = torch.tensor([[i, i] for i in range(len(features_adjusted))], dtype=torch.long).t().contiguous().to(device)
                        from .models import Data
                        dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)
                        with torch.no_grad():
                            reconstructed = model(dummy_data)
                            scores = torch.mean((dummy_x - reconstructed) ** 2, dim=1).cpu().numpy()
                        logger.debug("GNN Autoencoder scores range: [%.4f, %.4f]", np.min(scores), np.max(scores))
                    except Exception as e:
                        logger.warning("GNN model prediction failed for %s: %s, using fallback scores", model_name, e)
                        scores = np.random.random(len(features))
                else:
                    logger.warning("GNN model has no forward method for %s, using fallback scores", model_name)
                    scores = np.random.random(len(features))
            else:
                logger.warning("Unknown model method: %s for %s", method, model_name)
                scores = np.zeros(len(features))
            # Ensure scores are not all the same
            if np.all(scores == scores[0]):
                logger.warning("All scores are constant for %s, adding small noise", model_name)
                scores = scores + np.random.normal(0, 1e-6, size=scores.shape)
            # Always define predictions
            threshold_debug = np.percentile(scores, 90)
            predictions = (scores > threshold_debug).astype(int)
            logger.debug("Base model %s: %d anomalies detected out of %d samples (using debug threshold)", model_name, np.sum(predictions), len(predictions))
            return scores, predictions
            
        except Exception as e:
            logger.error("Error getting scores from %s: %s", model_name, e)
            return np.zeros(len(features)), np.zeros(len(features))
    
    def predict_anomalies(self, features: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using ensemble of base models.
        
        Args:
            features: Input features
            threshold: Detection threshold (if None, will be calculated)
        
        Returns:
            Tuple of (anomaly_scores, predictions)
        """
        logger.info("Ensemble prediction started with %d samples", len(features))
        
        if self.weights is None:
            logger.warning("Ensemble weights not optimized, using equal weights")
            self.weights = {model: 1.0 / len(self.base_models) for model in self.base_models.keys()}
        
        logger.debug("Ensemble weights: %s", self.weights)
        logger.debug("Available base models: %s", list(self.base_models.keys()))
        
        # Get scores from all base models
        base_scores = {}
        valid_models = []
        
        for model_name, weight in self.weights.items():
            if model_name in self.base_models and weight > 0:
                try:
                    logger.debug("Processing base model: %s (weight: %.4f)", model_name, weight)
                    model_data = self.base_models[model_name]
                    scores, _ = self._get_base_model_scores(model_data, features, model_name)
                    if scores is not None and len(scores) > 0:
                        base_scores[model_name] = scores
                        valid_models.append(model_name)
                        logger.debug("Successfully got scores from %s: shape=%s, range=[%.4f, %.4f]", 
                                   model_name, scores.shape, np.min(scores), np.max(scores))
                    else:
                        logger.warning("No valid scores from %s", model_name)
                except Exception as e:
                    logger.warning("Failed to get scores from %s: %s", model_name, e)
            else:
                logger.debug("Skipping %s: not in base_models or weight <= 0", model_name)
        
        logger.info("Valid base models for ensemble: %s", valid_models)
        
        if not valid_models:
            logger.error("No valid base models for ensemble prediction")
            return np.zeros(len(features)), np.zeros(len(features))
        
        # Combine scores using fusion method
        logger.debug("Using fusion method: %s", self.fusion_method)
        
        if self.fusion_method == 'weighted_average':
            ensemble_scores = np.zeros(len(features))
            total_weight = 0
            
            for model_name in valid_models:
                weight = self.weights[model_name]
                ensemble_scores += weight * base_scores[model_name]
                total_weight += weight
                logger.debug("Added %s contribution: weight=%.4f, score_range=[%.4f, %.4f]", 
                           model_name, weight, np.min(base_scores[model_name]), np.max(base_scores[model_name]))
            
            if total_weight > 0:
                ensemble_scores /= total_weight
                logger.debug("Weighted average computed with total_weight=%.4f", total_weight)
            else:
                ensemble_scores = np.mean(list(base_scores.values()), axis=0)
                logger.debug("Using unweighted average due to zero total_weight")
        
        elif self.fusion_method == 'weighted_median':
            # Calculate weighted median
            sorted_scores = np.sort(list(base_scores.values()), axis=0)
            weights_array = np.array([self.weights[model] for model in valid_models])
            weights_array = weights_array / weights_array.sum()
            
            # Calculate weighted median
            cumulative_weights = np.cumsum(weights_array)
            median_idx = np.searchsorted(cumulative_weights, 0.5)
            ensemble_scores = sorted_scores[median_idx]
            logger.debug("Weighted median computed with weights: %s", weights_array)
        
        elif self.fusion_method == 'max_voting':
            # Use maximum score from all models
            ensemble_scores = np.max(list(base_scores.values()), axis=0)
            logger.debug("Max voting computed from %d models", len(valid_models))
        
        else:
            logger.warning("Unknown fusion method: %s, using weighted average", self.fusion_method)
            ensemble_scores = np.mean(list(base_scores.values()), axis=0)
        
        logger.debug("Ensemble scores computed: range=[%.4f, %.4f]", np.min(ensemble_scores), np.max(ensemble_scores))
        
        # Apply confidence weighting if enabled
        if self.confidence_weighting and len(valid_models) > 1:
            logger.debug("Applying confidence weighting")
            # Calculate confidence based on agreement between models
            score_std = np.std(list(base_scores.values()), axis=0)
            confidence = 1.0 / (1.0 + score_std)  # Higher agreement = higher confidence
            ensemble_scores = ensemble_scores * confidence
            logger.debug("Confidence weighting applied: confidence_range=[%.4f, %.4f]", 
                        np.min(confidence), np.max(confidence))
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = np.percentile(ensemble_scores, 90)
            logger.debug("Calculated threshold: %.4f (90th percentile)", threshold)
        else:
            logger.debug("Using provided threshold: %.4f", threshold)
        
        # Make predictions
        predictions = (ensemble_scores > threshold).astype(int)
        
        logger.info("Ensemble prediction completed: %d anomalies detected out of %d samples", 
                   np.sum(predictions), len(predictions))
        logger.debug("Final ensemble scores: range=[%.4f, %.4f], threshold=%.4f", 
                    np.min(ensemble_scores), np.max(ensemble_scores), threshold)
        
        return ensemble_scores, predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble model."""
        return {
            'type': 'ensemble',
            'base_models': list(self.base_models.keys()),
            'weights': self.weights,
            'optimization_method': self.optimization_method,
            'fusion_method': self.fusion_method,
            'confidence_weighting': self.confidence_weighting
        }
