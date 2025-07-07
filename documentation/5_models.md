# Module: models.py

## Overview
This module implements the core machine learning models for unsupervised anomaly detection in AI agent trajectory data. It supports multiple model types, including:
- **Isolation Forest** (ensemble-based anomaly detection)
- **One-Class SVM** (support vector novelty detection)
- **GNN Diffusion Autoencoder** (deep graph neural network with reconstruction loss)
- **Ensemble Model** (combines multiple base models)

All models are trained in a purely unsupervised manner—no anomaly labels are used during training. The module is designed for extensibility, robust hyperparameter tuning, and integration with graph-based and tabular features.

---

## 1. Class and Function Reference

### `AnomalyDetectionModels`
- **Purpose:** Main class for training, tuning, and inference of all anomaly detection models.
- **Key Methods:**
  - `__init__(config: Dict)`
  - `prepare_training_data(features_df, embeddings=None, feature_columns=None)`
  - `train_isolation_forest(X, hyperparameter_tuning=True)`
  - `train_one_class_svm(X, hyperparameter_tuning=True)`
  - `train_gnn_autoencoder(graphs, hyperparameter_tuning=True)`
  - `train_ensemble_model(base_models, val_features, val_labels)`
  - `predict_anomalies(model_results, X_test, threshold=None)`
  - `save_models(models, filepath)` / `load_models(filepath)`
- **Internal Methods:**
  - `_tune_isolation_forest`, `_tune_one_class_svm`, `_tune_gnn_autoencoder`, `_convert_to_torch_geometric`, `_train_gnn_model`, etc.

### `GNNAutoencoder` (PyTorch nn.Module)
- **Purpose:** Encoder-decoder GNN for graph reconstruction and anomaly scoring.
- **Configurable:** GCN, GAT, GraphConv layers; hidden dims, dropout, etc.

### `EnsembleAnomalyDetector`
- **Purpose:** Combines base models using weighted average, median, or max voting. Optimizes weights on validation data.
- **Key Methods:**
  - `optimize_weights(val_features, val_labels)`
  - `predict_anomalies(features, threshold=None)`
  - `get_model_info()`

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                                      |
|--------------------------|---------|------------------------------------------------------------------|
| `n_estimators`           | int     | Number of trees in Isolation Forest                              |
| `contamination`          | float   | Expected anomaly proportion (IF)                                 |
| `max_features`           | float   | Max features per tree (IF)                                       |
| `max_samples`            | float   | Max samples per tree (IF)                                        |
| `kernel`                 | str     | SVM kernel type                                                  |
| `nu`                     | float   | SVM regularization parameter                                     |
| `gamma`                  | str     | SVM kernel coefficient                                           |
| `gnn_type`               | str     | GNN layer type (GCN, GAT, GraphConv)                             |
| `hidden_dims`            | list    | GNN hidden layer sizes                                           |
| `epochs`                 | int     | GNN training epochs                                              |
| `dropout`                | float   | Dropout rate for GNN                                             |
| `fusion_method`          | str     | Ensemble fusion method (weighted_average, weighted_median, max_voting) |
| `optimization_method`    | str     | Ensemble weight optimization method                              |

**Example config:**
```yaml
models:
  isolation_forest:
    n_estimators: [50, 100, 200]
    contamination: [0.05, 0.1, 0.2]
    max_features: [0.5, 1.0]
    max_samples: [0.5, 1.0]
    bootstrap: [false]
  one_class_svm:
    kernel: ["rbf", "linear"]
    nu: [0.05, 0.1, 0.2]
    gamma: ["scale", "auto"]
  gnn_autoencoder:
    gnn_type: ["GCN", "GAT", "GraphConv"]
    hidden_dims: [[64, 32], [128, 64]]
    epochs: [100, 200]
    dropout: [0.1, 0.3]
ensemble:
  fusion_method: "weighted_average"
  optimization_method: "validation_performance"
```

---

## 3. Workflow Pseudocode

```python
# Prepare training data
X, feature_columns = models.prepare_training_data(features_df, embeddings)

# Train models
if_result = models.train_isolation_forest(X, hyperparameter_tuning=True)
ocsvm_result = models.train_one_class_svm(X, hyperparameter_tuning=True)
gnn_result = models.train_gnn_autoencoder(graphs, hyperparameter_tuning=True)

# Train ensemble
ensemble_result = models.train_ensemble_model(
    base_models={
        'isolation_forest': if_result,
        'one_class_svm': ocsvm_result,
        'gnn_autoencoder': gnn_result
    },
    val_features=X_val,
    val_labels=y_val
)

# Predict anomalies
scores, preds = models.predict_anomalies(ensemble_result, X_test)
```

- **Each model** supports grid search for hyperparameters and returns a standardized result dict.
- **Ensemble** combines base model scores using optimized weights.
- **Embeddings** can be concatenated to features for all models.

---

## 4. Sample Output (Model Result)

```json
{
  "model": "IsolationForest(...) or GNNAutoencoder(...) or EnsembleAnomalyDetector(...) ",
  "best_params": {"n_estimators": 100, "contamination": 0.1, ...},
  "training_scores": [0.12, -0.03, ...],
  "method": "isolation_forest",
  "scaler": "StandardScaler(...) (if used)",
  "best_score": 0.87 (if available)
}
```

---

## 5. Data Structures and Integration

### Model Result Format
- All models return a dict with: `model`, `best_params`, `training_scores`, `method`, `scaler` (optional), `best_score` (optional)

### Embedding Integration
- Embeddings (Node2Vec, DeepWalk, GraphSAGE) are concatenated to tabular features for IF/SVM; GNN uses graph structure directly.

### Ensemble Model
- Combines base model scores using weighted average, median, or max voting; weights optimized on validation data.

### Model Saving/Loading
- Models can be saved/loaded via pickle for reproducibility.

---

## 6. Edge Cases and Error Handling
- **Missing values:** All models handle NaN/infinite values via imputation or scaling.
- **Small datasets:** Special logic for parameter selection and early stopping.
- **Inconsistent features:** Ensures feature columns are consistent between train/test splits.
- **GNN/torch errors:** Catches and logs errors, returns empty results if training fails.
- **Ensemble:** Skips invalid base models, normalizes weights, and logs warnings.

---

## 7. Integration Points
- **Input:** Takes features (DataFrame/numpy), embeddings (dict), and graphs (list) from previous modules.
- **Output:** Returns model result dicts, ensemble results, and predictions.
- **Downstream:** Used by evaluation, confidence analysis, and detailed trajectory analysis modules.
- **Config:** Reads from a config dict (YAML/JSON supported via utility functions).

---

## 8. Limitations and Concerns
- **Unsupervised Only:** No anomaly labels used during training; evaluation must be done separately.
- **Scalability:** GNN models may be slow for large graphs; SVMs may not scale to very large datasets.
- **Feature Engineering Dependency:** Model performance is highly dependent on quality and consistency of input features and embeddings.
- **Threshold Calibration:** Requires careful calibration on validation data for reliable anomaly detection.

---

## 9. Design Decisions
- **Extensibility:** Modular class structure allows easy addition of new models or ensemble strategies.
- **Robustness:** Handles missing values, inconsistent features, and small datasets with special logic.
- **Logging:** Extensive logging for training, tuning, and prediction steps.

---

## 10. Outcome
This module enables robust, extensible, and reproducible unsupervised anomaly detection for complex agent trajectory data, supporting both tabular and graph-based representations. It is a core component of the overall anomaly detection pipeline. 