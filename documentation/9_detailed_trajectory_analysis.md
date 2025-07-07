# 9. Detailed Trajectory Analysis Module (detailed_trajectory_analysis.py)

## Overview
This module provides detailed visualizations and analysis for individual agent trajectory graphs, supporting both normal and anomalous cases. It is designed to help interpret model predictions and understand trajectory structure and errors.

---

## 1. Class and Function Reference

### `DetailedTrajectoryAnalysis`
- **Purpose:** Main class for creating detailed visualizations and analysis of individual trajectories.
- **Key Methods:**
  - `__init__(output_dir: Path, format: str = 'png', dpi: int = 300, bbox_inches: str = 'tight')`
  - `create_detailed_trajectory_analysis(graphs, features_df, model_results, test_df) -> str`
  - `_create_simple_trajectory_analysis(graph, features, test_row, filename, trajectory_type) -> None`
  - `_create_single_trajectory_analysis(trajectory, features, model_results, test_row, filename, trajectory_type) -> None`
  - `_plot_trajectory_graph(graph, ax, trajectory_type) -> None`
  - `_plot_trajectory_steps_table(graph, ax, all_steps: bool = False) -> None`
  - `_plot_key_statistics(features, test_row, ax) -> None`
  - `_plot_model_predictions(features, model_results, ax) -> None`
  - `_plot_errors_table(graph, ax) -> None`
  - `_plot_node_type_distribution(graph, ax) -> None`

---

## 2. Configuration Parameters

| Parameter      | Type   | Description                        | Default   |
|----------------|--------|------------------------------------|-----------|
| `output_dir`   | Path   | Directory to save visualizations   | required  |
| `format`       | str    | Output image format (png, pdf, etc.) | 'png'     |
| `dpi`          | int    | Image resolution                   | 300       |
| `bbox_inches`  | str    | Bounding box for saving figures    | 'tight'   |

**Example Usage:**
```python
analyzer = DetailedTrajectoryAnalysis(output_dir=Path('charts/trajectories'))
analyzer.create_detailed_trajectory_analysis(graphs, features_df, model_results, test_df)
```

---

## 3. Workflow Pseudocode

```python
# Pseudocode for detailed trajectory analysis
analyzer = DetailedTrajectoryAnalysis(output_dir=Path('charts/trajectories'))
# 1. Select representative normal and anomalous trajectories
# 2. For each, generate a multi-panel figure with:
#    - Trajectory graph visualization
#    - Steps table
#    - Node type distribution
#    - Errors table
#    - Key statistics
#    - Model predictions
analyzer.create_detailed_trajectory_analysis(graphs, features_df, model_results, test_df)
```

---

## 4. Sample Output

- **normal_trajectory_1_detailed_analysis.png**: Multi-panel figure for a normal trajectory
- **anomalous_trajectory_1_detailed_analysis.png**: Multi-panel figure for an anomalous trajectory
- **Each figure includes:**
  - Trajectory graph (nodes/edges, colored by type)
  - Steps table (node/agent types, descriptions)
  - Node type distribution (bar chart)
  - Errors table (failed/error nodes)
  - Key statistics (length, speed, duration, complexity, labels)
  - Model predictions (bar chart of anomaly scores)

---

## 5. Data Structures and Integration Points

- **Input:**
  - `graphs`: List[nx.DiGraph] (trajectory graphs)
  - `features_df`: pd.DataFrame (features for each trajectory)
  - `model_results`: Dict[str, Any] (model outputs)
  - `test_df`: pd.DataFrame (labels and metadata)
- **Output:**
  - PNG figures in `output_dir`
- **Integration:**
  - Consumes outputs from graph processing, feature engineering, and models modules
  - Used for model debugging, error analysis, and presentation

---

## 6. Edge Cases and Error Handling
- Handles empty or invalid graphs gracefully (annotates as empty/error)
- Skips missing data or features with clear logging
- Robust to plotting errors (continues with other trajectories)
- Limits number of trajectories visualized for clarity

---

## 7. Limitations and Concerns
- Visualizations are designed for a small number of trajectories; not suitable for large-scale batch analysis
- Some plots (e.g., model predictions) may require domain knowledge to interpret
- Poorly constructed graphs or missing features may reduce visualization quality

---

## 8. Design Decisions
- Multi-panel layout for holistic view of each trajectory
- Publication-quality plots with consistent styling
- Modular plotting functions for extensibility

---

## 9. Results and Observations

### Key Insights from Detailed Trajectory Analysis

- **Structural Patterns:** Normal trajectories typically show clear, linear or branched execution flows, while anomalous trajectories often exhibit irregularities such as excessive branching, cycles, or disconnected nodes.
- **Error Localization:** Error tables and node coloring make it easy to pinpoint where failures or anomalies occur within a trajectory, supporting rapid debugging and root cause analysis.
- **Node and Agent Diversity:** The analysis highlights the diversity of node and agent types involved in each trajectory, with anomalous cases often involving more handoffs, tool calls, or error/recovery nodes.
- **Model Prediction Interpretation:** Bar charts of model anomaly scores provide immediate visual feedback on which models are most sensitive to a given trajectory's structure or errors.
- **Statistical Context:** Key statistics (length, speed, duration, complexity) help contextualize each trajectory, revealing outliers or edge cases.

### Observations about the Visualization Module

- **Strengths:**
  - Enables deep, case-by-case understanding of both normal and anomalous agent behavior.
  - Highly customizable and extensible for new plot types or statistics.
  - Robust to missing or malformed data, with clear error handling and logging.
- **Limitations:**
  - Not intended for large-scale batch analysis; best used for selected, representative cases.
  - Some visualizations (e.g., model predictions) may require explanation for non-technical audiences.
- **Recommendations:**
  - Use in conjunction with aggregate analysis modules for a complete picture.
  - Select trajectories that are representative of key behaviors or failure modes for maximum insight.

---

## 10. Outcome
This module enables detailed, interpretable, and visually rich analysis of agent trajectories, supporting both model debugging and presentation to stakeholders. 