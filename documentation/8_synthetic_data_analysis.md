# 8. Synthetic Data Analysis Module (synthetic_data_analysis.py)

## Overview
This module provides comprehensive analysis and visualization of synthetic agent trajectory data, comparing normal and anomalous trajectories across multiple dimensions. It is designed to support both model development and presentation.

---

## 1. Class and Function Reference

### `SyntheticDataAnalyzer`
- **Purpose:** Main class for loading, analyzing, and visualizing synthetic trajectory data.
- **Key Methods:**
  - `__init__(results_dir: str = "results")`
  - `create_comprehensive_analysis()`
  - `_plot_agent_type_distribution(ax)`
  - `_plot_tool_usage_patterns(ax)`
  - `_plot_node_type_distribution(ax)`
  - `_plot_trajectory_metrics(ax)`
  - `_plot_handoff_patterns(ax)`
  - `_plot_llm_call_patterns(ax)`
  - `_plot_planning_patterns(ax)`
  - `_plot_error_patterns(ax)`
  - `_plot_success_metrics(ax)`
  - `_create_detailed_agent_analysis()`
  - `_create_detailed_tool_analysis()`
  - `_create_detailed_workflow_analysis()`
  - `_create_statistical_summary()`
  - `_load_pickle(filepath: Path)`

### Script Entry Point
- `main()`: Instantiates `SyntheticDataAnalyzer` and runs the full analysis pipeline.

---

## 2. Configuration Parameters

| Parameter         | Type   | Description                                 | Default   |
|-------------------|--------|---------------------------------------------|-----------|
| `results_dir`     | str    | Directory containing data and output folders | 'results' |
| `charts_dir`      | str    | Directory for saving generated charts        | 'results/charts' |
| `data_dir`        | str    | Directory for input data (CSV, PKL)         | 'results/data'   |

**Example Usage:**
```python
analyzer = SyntheticDataAnalyzer(results_dir="results")
analyzer.create_comprehensive_analysis()
```

---

## 3. Workflow Pseudocode

```python
# Pseudocode for comprehensive synthetic data analysis
analyzer = SyntheticDataAnalyzer(results_dir="results")
analyzer.create_comprehensive_analysis()
# This will:
# 1. Load features and trajectory pickles
# 2. Split into normal/anomalous
# 3. Generate 3x3 grid of comparative plots
# 4. Create detailed agent, tool, and workflow analyses
# 5. Output statistical summary CSV and plots
```

---

## 4. Sample Output

- **comprehensive_synthetic_data_analysis.png**: 3x3 grid of comparative plots (agent, tool, node, trajectory, handoff, LLM, planning, error, success)
- **detailed_agent_analysis.png**: Pie, bar, and heatmap plots for agent types
- **detailed_tool_analysis.png**: Bar, scatter, and heatmap plots for tool usage
- **detailed_workflow_analysis.png**: Bar, scatter, and heatmap plots for workflow metrics
- **statistical_summary.png**: Bar plots for mean and percent difference of key metrics
- **synthetic_data_statistical_summary.csv**: Tabular summary of all key metrics (mean, std, min, max, percent diff)

---

## 5. Data Structures and Integration Points

- **Input:**
  - `features_df`: pd.DataFrame (from CSV, with all extracted features)
  - `normal_trajectories`, `anomalous_trajectories`: List[AgentTrajectory] (from PKL)
- **Output:**
  - PNG figures in `charts_dir`
  - CSV summary in `charts_dir`
- **Integration:**
  - Consumes outputs from data generation and feature extraction modules
  - Used for model development, debugging, and presentation

---

## 6. Edge Cases and Error Handling
- Handles missing or corrupt pickle/CSV files gracefully (prints warnings, skips analysis)
- Skips plotting if required columns are missing
- Robust to empty or small datasets (plots will show available data)
- All plotting errors are caught and reported

---

## 7. Limitations and Concerns
- Analysis is limited to synthetic data; real-world generalization may require further validation
- Designed for moderate dataset sizes; very large datasets may require optimization
- Some plots may require domain knowledge to interpret

---

## 8. Design Decisions
- Multi-panel layout for holistic, comparative analysis
- Exportable results (CSV, PNG) for integration with other tools
- Modular plotting functions for extensibility

---

## 9. Outcome
This module enables detailed, interpretable, and visually rich analysis of synthetic agent trajectory data, supporting both model development and effective communication of results.

---

## 10. Results and Observations

### Key Findings from Synthetic Data Analysis

- **Diversity of Patterns:** The synthetic data generator successfully produces a wide range of trajectory patterns, including simple linear, branched, multi-agent handoff, complex research, and error recovery workflows. This diversity is reflected in the distribution plots for node types, agent types, and tool usage.

- **Realism and Workflow Complexity:** Generated trajectories exhibit realistic workflow characteristics, such as varying trajectory lengths, branching factors, and agent/tool handoff patterns. The inclusion of timing variance, error injection, and recovery nodes adds to the authenticity of the data.

- **Normal vs. Anomalous Separation:** Comparative plots (e.g., trajectory length, error frequency, handoff count) show clear statistical differences between normal and anomalous trajectories. Anomalous data typically exhibits higher error counts, more complex handoff patterns, and lower completion/success rates, validating the effectiveness of the anomaly injection logic.

- **Statistical Summary:** The statistical summary (see `statistical_summary.png` and `synthetic_data_statistical_summary.csv`) quantifies these differences, with percent differences in key metrics (e.g., error count, handoff frequency, LLM call ratio) often exceeding 50-100% between normal and anomalous cases.

- **Agent and Tool Usage:** The generator produces a balanced mix of agent and tool types, with some tools/agents more prevalent in anomalous scenarios (e.g., error-handling tools, recovery agents). Correlation heatmaps reveal strong associations between certain agent/tool types and anomaly status.

- **Success and Completion Rates:** Normal trajectories consistently achieve higher success and completion rates, while anomalous trajectories show a marked drop, as expected from the injected anomalies and workflow disruptions.

### Observations about the Synthetic Data Generator

- **Strengths:**
  - Highly configurable and extensible, supporting new patterns and agent/tool types with minimal code changes.
  - Reproducible results via random seed control.
  - Realistic variance and error modeling, supporting robust model training and evaluation.

- **Limitations:**
  - As with all synthetic data, some real-world nuances (e.g., rare edge cases, human error patterns) may not be fully captured.
  - Parameter sensitivity: The diversity and realism of the data depend on careful tuning of pattern weights, node counts, and error probabilities.
  - Scalability: Very large datasets or highly complex trajectories may require optimization for speed and memory.

- **Recommendations:**
  - For best results, periodically validate synthetic data against real-world samples and adjust configuration as needed.
  - Use the statistical summary and comparative plots to guide further refinement of the generator and anomaly injection logic.

--- 