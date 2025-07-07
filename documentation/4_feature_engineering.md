# Module: feature_engineering.py

## Overview
This module is responsible for extracting comprehensive features from agent trajectory graphs for use in anomaly detection models. It supports multiple feature categories, robust cleaning, and detailed feature analysis.

---

## 1. Class and Function Reference

### `FeatureExtractor`
- **Purpose:** Main class for extracting structural, temporal, semantic, and DAG features from trajectory graphs.
- **Key Methods:**
  - `__init__(config: Dict)`
  - `extract_features(graphs: List[nx.DiGraph]) -> pd.DataFrame`
  - `get_feature_importance_analysis(features_df: pd.DataFrame) -> Dict[str, Any]`
- **Internal Methods:**
  - `_extract_graph_features`, `_extract_metadata_features`, `_extract_consolidated_structural_features`, `_extract_structural_features`, `_extract_dag_features`, `_extract_temporal_features`, `_extract_semantic_features`, `_extract_degree_features`, `_extract_connectivity_features`, `_extract_centrality_features`, `_extract_topology_features`, `_extract_path_features`, `_extract_adjacency_features`, `_clean_and_validate_features`, `_log_feature_statistics`, `_safe_calculation`, `_get_empty_features`, etc.

### `FeatureConfig`
- **Purpose:** Holds configuration constants for feature engineering (imputation, thresholds, feature categories, etc.)

---

## 2. Configuration Parameters

| Parameter                        | Type    | Description                                                      |
|-----------------------------------|---------|------------------------------------------------------------------|
| `structural_features`             | list    | List of structural features to extract                           |
| `dag_features`                    | list    | List of DAG-specific features to extract                         |
| `temporal_features`               | list    | List of temporal features to extract                             |
| `semantic_features`               | list    | List of semantic features to extract                             |
| `INF_REPLACEMENT_POSITIVE`        | float   | Value to replace positive infinity                               |
| `INF_REPLACEMENT_NEGATIVE`        | float   | Value to replace negative infinity                               |
| `MISSING_VALUE_THRESHOLD`         | float   | Threshold for median imputation                                  |
| `HIGH_MISSING_RATE_THRESHOLD`     | float   | Threshold for zero imputation                                    |
| `EXECUTION_GAP_THRESHOLD`         | float   | Threshold for detecting execution gaps (seconds)                 |
| `TEMPORAL_ANOMALY_Z_SCORE`        | float   | Z-score threshold for temporal anomaly detection                 |

**Example config:**
```yaml
feature_engineering:
  structural_features: ['num_nodes', 'num_edges', 'density', 'diameter', 'clustering_coefficient', 'transitivity', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']
  dag_features: ['is_dag', 'topological_sort_validation', 'longest_path_in_dag', 'dag_depth', 'nodes_per_level', 'branching_factor', 'merge_points', 'parallel_paths']
  temporal_features: ['total_duration', 'average_node_duration', 'duration_variance', 'inter_node_delays']
  semantic_features: ['agent_type_counts', 'handoff_frequency', 'tool_type_distribution', 'error_frequency']
```

---

## 3. Workflow Pseudocode

```python
# Extract features from graphs
features_df = extractor.extract_features(graphs)

# Analyze feature importance and distributions
analysis = extractor.get_feature_importance_analysis(features_df)
```

- **Feature extraction** is performed in a single pass per graph for efficiency.
- **Cleaning and validation** handle missing, infinite, and non-numeric values.
- **Feature importance analysis** groups features by category and computes statistics.

---

## 4. Sample Output (Features DataFrame)

```json
{
  "graph_id": "graph_0",
  "num_nodes": 12,
  "num_edges": 15,
  "density": 0.23,
  "diameter": 5,
  "clustering_coefficient": 0.12,
  "transitivity": 0.09,
  "betweenness_centrality_mean": 0.03,
  "closeness_centrality_mean": 0.21,
  "eigenvector_centrality_mean": 0.18,
  "is_dag": true,
  "longest_path_in_dag": 7,
  "dag_depth": 4,
  "total_duration": 120.5,
  "average_node_duration": 10.2,
  "agent_type_counts": {"PLANNER": 3, "EXECUTOR": 7, "VALIDATOR": 2},
  ...
}
```

---

## 5. Feature Categories and Data Structures

### Structural Features
- `num_nodes`, `num_edges`, `density`, `diameter`, `clustering_coefficient`, `transitivity`, `betweenness_centrality_mean`, `closeness_centrality_mean`, `eigenvector_centrality_mean`, `degree_assortativity`, `radius`, `average_shortest_path_length`, etc.

### DAG Features
- `is_dag`, `topological_sort_validation`, `longest_path_in_dag`, `dag_depth`, `nodes_per_level_mean`, `branching_factor`, `merge_points`, `parallel_paths`, etc.

### Temporal Features
- `total_duration`, `average_node_duration`, `duration_variance`, `inter_node_delays`, `duration_min`, `duration_max`, `duration_median`, etc.

### Semantic Features
- `agent_type_counts`, `handoff_frequency`, `tool_type_distribution`, `error_frequency`, etc.

### Cleaning/Imputation
- Infinite values replaced with `INF_REPLACEMENT_POSITIVE`/`NEGATIVE`
- NaN/missing values imputed by median or zero, depending on missing rate

---

## 6. Edge Cases and Error Handling
- **Empty graphs:** Returns zero/empty features for all categories.
- **Disconnected graphs:** Uses fallback values for diameter/path features.
- **Missing/infinite values:** Replaced or imputed according to config.
- **Non-numeric columns:** Skipped or logged during cleaning.

---

## 7. Integration Points
- **Input:** Takes a list of graphs (NetworkX DiGraph) from graph processing.
- **Output:** Returns a pandas DataFrame of features (one row per graph).
- **Downstream:** Used by modeling, evaluation, and analysis modules.
- **Config:** Reads from a config dict (YAML/JSON supported via utility functions).

---

## 8. Limitations and Concerns
- **Feature Redundancy:** Some features may be highly correlated or redundant; feature selection may be needed.
- **Scalability:** Large graphs or many features may increase computation time and memory usage.
- **Dependency on Graph Quality:** Poorly constructed graphs yield less informative features.
- **Imputation Choices:** Imputation strategies may affect downstream model performance.

---

## 9. Design Decisions
- **Single-Pass Extraction:** Consolidated feature extraction for efficiency and consistency.
- **Extensibility:** Modular design allows easy addition of new feature types or categories.

---

## 10. Outcome
This module provides a robust, extensible, and comprehensive feature extraction pipeline for agent trajectory graphs, enabling effective anomaly detection and model interpretability. 