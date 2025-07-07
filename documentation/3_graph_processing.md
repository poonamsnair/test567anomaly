# Module: graph_processing.py

## Overview
This module handles the conversion of agent trajectories into graph representations and the generation of graph embeddings for downstream anomaly detection. It supports multiple embedding algorithms and comprehensive hyperparameter tuning.

---

## 1. Class and Function Reference

### `GraphProcessor`
- **Purpose:** Main class for converting agent trajectories to graphs and generating embeddings/statistics.
- **Key Methods:**
  - `__init__(config: Dict)`
  - `trajectories_to_graphs(trajectories: List[AgentTrajectory]) -> List[nx.DiGraph]`
  - `generate_node2vec_embeddings(graphs: List[nx.DiGraph], hyperparameter_tuning: bool = True) -> Dict[str, Any]`
  - `generate_deepwalk_embeddings(graphs: List[nx.DiGraph], hyperparameter_tuning: bool = True) -> Dict[str, Any]`
  - `generate_graphsage_embeddings(graphs: List[nx.DiGraph], hyperparameter_tuning: bool = True) -> Dict[str, Any]`
  - `get_graph_statistics(graphs: List[nx.DiGraph]) -> Dict[str, Any]`
  - `save_embeddings(embeddings_data: Dict[str, Any], filepath: str) -> None`
  - `load_embeddings(filepath: str) -> Dict[str, Any]`
  - `load_graphs(filepath: str) -> List[nx.DiGraph]`

#### Internal Methods
- `_trajectory_to_graph`, `_extract_node_attributes`, `_extract_edge_attributes`, `_tune_*_hyperparameters`, `_generate_*_with_params`, `_aggregate_node_embeddings`, `_calculate_centralities`, `_evaluate_embeddings`, `_convert_to_torch_geometric`, `_train_graphsage_model`

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                                      |
|--------------------------|---------|------------------------------------------------------------------|
| `node2vec`               | dict    | Node2Vec hyperparameters (dimensions, walk_length, num_walks, etc.) |
| `deepwalk`               | dict    | DeepWalk hyperparameters (dimensions, walk_length, num_walks, etc.) |
| `graphsage`              | dict    | GraphSAGE hyperparameters (hidden_dims, output_dim, etc.)        |
| `aggregation_methods`    | list    | Methods for aggregating node embeddings (mean, max, weighted_mean, etc.) |

**Example config:**
```yaml
graph_processing:
  node2vec:
    dimensions: [64, 128]
    walk_length: [30, 50]
    num_walks: [200, 500]
    p: [0.5, 1.0]
    q: [0.5, 1.0]
    window: [5, 10]
    min_count: [1]
  deepwalk:
    dimensions: [64, 128]
    walk_length: [40, 80]
    num_walks: [100, 200]
    window: [5, 10]
    min_count: [1]
  graphsage:
    hidden_dims: [[64, 128]]
    output_dim: [64]
    learning_rate: [0.001]
    epochs: [100]
    batch_size: [32]
    dropout: [0.1]
    aggregator: ['mean', 'max']
  aggregation_methods: ['mean', 'weighted_mean']
```

---

## 3. Workflow Pseudocode

```python
# Convert trajectories to graphs
graphs = processor.trajectories_to_graphs(trajectories)

# Generate embeddings (Node2Vec, DeepWalk, GraphSAGE)
node2vec_result = processor.generate_node2vec_embeddings(graphs, hyperparameter_tuning=True)
deepwalk_result = processor.generate_deepwalk_embeddings(graphs, hyperparameter_tuning=True)
graphsage_result = processor.generate_graphsage_embeddings(graphs, hyperparameter_tuning=True)

# Get statistics
stats = processor.get_graph_statistics(graphs)
```

- **Each embedding method** supports grid search for hyperparameters and returns best params, embeddings, and scores.
- **Node/edge attributes** are extracted for each graph for downstream feature engineering.
- **Persistence**: Embeddings and graphs can be saved/loaded via pickle.

---

## 4. Sample Output (Graph and Embedding JSON)

```json
{
  "graph": {
    "trajectory_id": "uuid1",
    "nodes": [
      {"node_id": "n1", "node_type": "TOOL_CALL", "agent_type": "EXECUTOR", ...},
      {"node_id": "n2", "node_type": "REASONING", ...},
      ...
    ],
    "edges": [
      {"source_node_id": "n1", "target_node_id": "n2", "edge_type": "execution_flow", ...},
      ...
    ],
    "is_anomalous": false,
    "completion_rate": 1.0
  },
  "embeddings": {
    "node2vec": {"graph_0": [0.1, 0.2, ...]},
    "deepwalk": {"graph_0": [0.3, 0.4, ...]},
    "graphsage": {"graph_0": [0.5, 0.6, ...]}
  },
  "statistics": {
    "node_count": {"min": 5, "max": 20, "mean": 12.3, "std": 3.2},
    "edge_count": {"min": 4, "max": 19, "mean": 11.7, "std": 3.1},
    ...
  }
}
```

---

## 5. Data Structures

### Node Attributes
- `node_type`, `agent_type`, `duration`, `status`, `performance_score`, `cpu_usage`, `memory_usage`, `network_calls`, `retry_count`, `is_failed`, `success`, `is_anomalous`, `anomaly_type`, `anomaly_severity`, `tool_type`, `tool_success`, `memory_operation`, `planning_type`, `reasoning_type`, `validation_result`, `planning_confidence`, `reasoning_confidence`, `validation_score`, `source_agent`, `target_agent`, `handoff_success`, `start_timestamp`

### Edge Attributes
- `edge_type`, `relationship_type`, `latency`, `probability`, `confidence`, `success_rate`, `error_count`, `timeout_count`, `weight`, `is_anomalous`, `reliability_score`, `data_size`

### Embedding Formats
- **Node2Vec/DeepWalk:** `{graph_id: np.ndarray}`
- **GraphSAGE:** `{graph_id: np.ndarray}`
- **All embeddings** are graph-level (aggregated from node embeddings)

### Statistics
- `node_count`, `edge_count`, `density`, `node_types`, `edge_types`, `anomalous_graphs`, `avg_path_length`

---

## 6. Edge Cases and Error Handling
- **Empty/small graphs:** Returns zero embeddings; skips or logs warnings.
- **Missing attributes:** Defaults or skips missing values; logs warnings.
- **PyTorch Geometric missing:** Skips GraphSAGE and logs a warning.
- **Embedding failures:** Catches exceptions, logs, and returns zero vectors.

---

## 7. Integration Points
- **Input:** Takes a list of `AgentTrajectory` objects (from data generation/anomaly injection).
- **Output:** Returns graphs (NetworkX DiGraph), embeddings (dict), and statistics (dict).
- **Downstream:** Used by feature engineering and modeling modules.
- **Persistence:** Embeddings and graphs can be saved/loaded for reproducibility.

---

## 8. Limitations and Concerns
- **Dependency on PyTorch Geometric:** GraphSAGE requires this library; if unavailable, falls back to shallow methods.
- **Embedding Quality:** Highly dependent on graph structure and feature engineering; poor graphs yield poor embeddings.
- **Scalability:** Large graphs or many graphs may slow down embedding generation, especially for deep models.
- **Parameter Sensitivity:** Embedding quality and downstream performance are sensitive to hyperparameter choices.

---

## 9. Design Decisions
- **Extensibility:** Modular class structure allows easy addition of new embedding methods or graph statistics.
- **Error Handling:** Extensive logging and error handling for all conversion and embedding steps.

---

## 10. Outcome
This module enables the transformation of complex agent trajectories into rich graph representations and embeddings, supporting advanced anomaly detection and graph-based machine learning workflows. 