# Feature Embedding for Agent Trajectory Graphs: GraphSAGE, DeepWalk, Node2Vec

## Overview
Graph-based feature embedding is central to representing agent trajectories for downstream anomaly detection and machine learning. This document details the three main embedding methods used in the system—**Node2Vec**, **DeepWalk**, and **GraphSAGE**—and explains how their outputs are integrated into the modeling pipeline.

---

## 1. Embedding Methods: Technical Comparison

| Method      | Type         | Core Idea                        | Input         | Output         | Strengths                        | Limitations                  |
|-------------|--------------|----------------------------------|---------------|---------------|----------------------------------|------------------------------|
| Node2Vec    | Shallow      | Biased random walks + skip-gram  | Graph         | Node vectors   | Fast, tunable, interpretable     | Ignores node features        |
| DeepWalk    | Shallow      | Uniform random walks + skip-gram | Graph         | Node vectors   | Simple, robust, scalable         | Ignores node features        |
| GraphSAGE   | Deep (GNN)   | Aggregates neighbor features     | Graph + feats | Node vectors   | Uses node features, inductive    | Needs PyTorch Geometric, slower |

- **Node2Vec**: Learns node embeddings by simulating biased random walks and applying the skip-gram model. Tunable parameters (walk length, p, q) control exploration/exploitation.
- **DeepWalk**: Similar to Node2Vec but uses uniform random walks. Simpler, less tunable, but robust for many graph types.
- **GraphSAGE**: A graph neural network that learns node embeddings by aggregating features from local neighborhoods. Supports mean, max, and weighted aggregators. Can use node attributes (e.g., type, performance, error status).

---

## 2. Technical Workflow

```python
# 1. Convert agent trajectories to NetworkX graphs
graphs = processor.trajectories_to_graphs(trajectories)

# 2. Generate node embeddings
node2vec_result = processor.generate_node2vec_embeddings(graphs, hyperparameter_tuning=True)
deepwalk_result = processor.generate_deepwalk_embeddings(graphs, hyperparameter_tuning=True)
graphsage_result = processor.generate_graphsage_embeddings(graphs, hyperparameter_tuning=True)

# 3. Aggregate node embeddings to graph-level features (mean, max, weighted)
# 4. Pass graph-level embeddings to downstream models (e.g., Isolation Forest, GNN Autoencoder, Ensemble)
```

- **Hyperparameter tuning** is performed for each method (grid search over dimensions, walk length, etc.).
- **Aggregation**: Node embeddings are aggregated (mean, max, weighted mean) to produce a single vector per graph for use in tabular models.
- **Persistence**: Embeddings can be saved/loaded for reproducibility.

---

## 3. Integration with Downstream Models

- **Tabular Models (Isolation Forest, One-Class SVM, etc.):**
  - Use aggregated graph embeddings as input features, optionally concatenated with engineered features.
- **GNN Models (GraphSAGE Autoencoder):**
  - Use node-level embeddings directly for reconstruction and anomaly scoring.
- **Ensemble Models:**
  - Combine predictions from models using different embedding types for robust anomaly detection.

**Example:**
- Node2Vec/DeepWalk embeddings → mean aggregation → Isolation Forest
- GraphSAGE node embeddings → autoencoder → reconstruction loss as anomaly score
- All embeddings → ensemble model for final prediction

---

## 4. Strengths and Limitations

- **Strengths:**
  - Multiple embedding methods capture both structural and attribute-based information
  - Hyperparameter tuning enables adaptation to different graph types
  - Embeddings are modular and reusable across models
- **Limitations:**
  - Shallow methods (Node2Vec, DeepWalk) ignore node features
  - GraphSAGE requires more computation and PyTorch Geometric
  - Embedding quality is sensitive to graph construction and feature engineering

---

## 5. Practical Recommendations

- Use **Node2Vec** or **DeepWalk** for fast, scalable embedding of large or feature-poor graphs
- Use **GraphSAGE** when node attributes are rich and inductive generalization is needed
- Always aggregate node embeddings for tabular models; use node-level embeddings for GNNs
- Tune hyperparameters for each dataset; monitor embedding quality with downstream model performance
- Save and version embeddings for reproducibility and debugging

---

## 6. Outcome
Feature embedding transforms complex agent trajectory graphs into dense, informative vectors, enabling advanced anomaly detection and robust machine learning workflows across the system.

---

## 7. Experimental Results and Model Performance (Fast Test, July 2025)

### Overview
- **Synthetic dataset:** 1,000 normal + 250 anomalous agent trajectories
- **Feature extraction:** 214 features (structural, semantic, temporal, DAG-based)
- **Embeddings:** Node2Vec, DeepWalk, GraphSAGE (with reduced dimensions for speed)
- **Models evaluated:** Isolation Forest, One-Class SVM, GNN Autoencoder, Ensemble
- **Checkpointing:** Enabled for embeddings and model training

### Key Results
| Model             | F1 Score | Precision | Recall | AUC-ROC |
|-------------------|---------:|----------:|-------:|--------:|
| Isolation Forest  |   0.547  |   0.667   | 0.464  |  0.752  |
| One-Class SVM     |   0.615  |   0.532   | 0.728  |  0.765  |
| GNN Autoencoder   |   0.556  |   0.385   | 1.000  |  0.605  |
| Ensemble Model    |   0.556  |   0.385   | 1.000  |  0.631  |

- **Best model:** One-Class SVM (F1: 0.615)
- **See:** `results_fast_test/charts/model_performance_comparison.png` for annotated bar plot

### Visualizations
- Embedding t-SNE: `results_fast_test/charts/embedding_tsne_visualization.png`
- Model performance: `results_fast_test/charts/model_performance_comparison.png`
- ROC/PR curves: `results_fast_test/charts/roc_curves.png`, `results_fast_test/charts/precision_recall_curves.png`
- Feature importance: `results_fast_test/charts/feature_importance.png`

### Recommendations
- **One-Class SVM** is recommended for deployment on similar synthetic data (best F1, balanced precision/recall)
- **Isolation Forest**: Higher precision, but lower recall (misses anomalies)
- **GNN/Ensemble**: High recall, but low precision (many false positives)
- For production, further tuning and real-world validation are advised

### Notes
- All results are reproducible with `config_fast_test.yaml` and checkpointing enabled
- For details, see the full analysis report: `results_fast_test/reports/analysis_report.md`

--- 