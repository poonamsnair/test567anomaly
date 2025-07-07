# Module: data_generation.py

## Overview
This module generates synthetic agent trajectories for anomaly detection, supporting a variety of realistic execution patterns and workflow complexities. It is foundational for training, evaluation, and benchmarking of anomaly detection models.

---

## 1. Class and Function Reference

### `TrajectoryGenerator`
- **Purpose:** Main class for generating synthetic agent trajectories with configurable patterns, node/edge types, and metadata.
- **Key Methods:**
  - `__init__(config: Dict, random_seed: int = 42)`
  - `generate_trajectories(num_trajectories: int) -> List[AgentTrajectory]`
  - `_generate_simple_linear_trajectory() -> AgentTrajectory`
  - `_generate_branched_analysis_trajectory() -> AgentTrajectory`
  - `_generate_multi_agent_handoff_trajectory() -> AgentTrajectory`
  - `_generate_complex_research_trajectory() -> AgentTrajectory`
  - `_generate_error_recovery_trajectory() -> AgentTrajectory`
  - `_create_*_node(...) -> TrajectoryNode` (user enquiry, planning, reasoning, tool call, LLM call, validation, handoff, memory, observation, action)
  - `_create_*_edge(...) -> TrajectoryEdge` (execution, handoff, data dependency)
  - `_add_trajectory_variance(trajectory: AgentTrajectory) -> None`

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                                      |
|--------------------------|---------|------------------------------------------------------------------|
| `random_seed`            | int     | Seed for reproducibility                                         |
| `trajectory_patterns`    | dict    | Dict of pattern configs (min/max nodes, weights, etc.)           |
| `min_nodes`/`max_nodes`  | int     | Node count range for each pattern                                |
| `weight`                 | float   | Probability weight for pattern selection                         |
| `task_templates`         | dict    | Task description templates by category                           |
| `tool_transitions`       | dict    | Probabilities for tool usage transitions                         |

**Example config:**
```yaml
random_seed: 42
data_generation:
  trajectory_patterns:
    simple_linear: {min_nodes: 5, max_nodes: 10, weight: 0.2}
    branched_analysis: {min_nodes: 10, max_nodes: 20, weight: 0.2}
    multi_agent_handoffs: {min_nodes: 15, max_nodes: 25, weight: 0.2}
    complex_research: {min_nodes: 20, max_nodes: 40, weight: 0.2}
    error_recovery: {min_nodes: 10, max_nodes: 30, weight: 0.2}
```

---

## 3. Workflow Pseudocode

```python
# Main generation loop
for i in range(num_trajectories):
    pattern = select_pattern_by_weight(config['trajectory_patterns'])
    if pattern == 'simple_linear':
        traj = generate_simple_linear_trajectory()
    elif pattern == 'branched_analysis':
        traj = generate_branched_analysis_trajectory()
    ...
    add_trajectory_variance(traj)
    traj.calculate_metrics()
    trajectories.append(traj)
```

- **Pattern-specific methods** create nodes/edges according to the workflow logic, e.g.:
  - User enquiry → Planning → Execution (tool/LLM/reasoning) → Action
  - Branches, handoffs, error/recovery, etc.
- **Variance injection** adds random timing, performance, and error attributes.
- **Post-processing** ensures all nodes are connected and metrics are up to date.

---

## 4. Sample Output (Trajectory JSON)

```json
{
  "task_description": "Analyze customer satisfaction data from Q3 surveys",
  "nodes": [
    {"node_id": "uuid1", "node_type": "USER_ENQUIRY", "start_time": "2024-06-01T10:00:00", ...},
    {"node_id": "uuid2", "node_type": "PLANNING", "planning_type": "task_decomposition", ...},
    {"node_id": "uuid3", "node_type": "TOOL_CALL", "tool_type": "ANALYZE_DATA", ...},
    ...
  ],
  "edges": [
    {"source_node_id": "uuid1", "target_node_id": "uuid2", "edge_type": "execution_flow", ...},
    {"source_node_id": "uuid2", "target_node_id": "uuid3", "edge_type": "execution_flow", ...},
    ...
  ],
  "metadata": {
    "agent_types": ["PLANNER", "EXECUTOR"],
    "tool_usage": ["ANALYZE_DATA"],
    "is_anomalous": false,
    "completion_rate": 1.0
  }
}
```

---

## 5. Data Structures

### Node Types (`NodeType`)
- `USER_ENQUIRY`, `PLANNING`, `REASONING`, `TOOL_CALL`, `LLM_CALL`, `VALIDATION`, `HANDOFF`, `MEMORY_ACCESS`, `OBSERVATION`, `ACTION`

### Edge Types
- `execution_flow`, `handoff`, `data_dependency`

### Node Attributes
- `node_id`, `node_type`, `agent_type`, `start_time`, `end_time`, `duration`, `description`, `status`, `error_code`, `error_message`, `tool_type`, `planning_type`, `reasoning_type`, `validation_criteria`, `memory_operation`, `observation_source`, etc.

### Edge Attributes
- `source_node_id`, `target_node_id`, `edge_type`, `relationship_type`, `creation_time`, `latency`, `probability`, `confidence`, `data_transferred`

---

## 6. Edge Cases and Error Handling
- **Disconnected nodes:** Post-processing ensures all nodes are connected; warnings are logged and linear chains are created if needed.
- **Empty patterns:** Defaults to simple linear if an unknown pattern is selected.
- **Failed tool calls:** Nodes are marked with error codes/messages; recovery nodes are added in error recovery patterns.
- **Randomization:** All random choices are seeded for reproducibility.

---

## 7. Integration Points
- **Output:** Returns a list of `AgentTrajectory` objects (can be serialized to JSON/pickle).
- **Downstream:** Used by anomaly injection, graph processing, and feature engineering modules.
- **Config:** Reads from a config dict (YAML/JSON supported via utility functions).

---

## 8. Limitations and Concerns
- **Synthetic Nature:** Generated data may not capture all real-world nuances; further validation may be needed for deployment.
- **Parameter Sensitivity:** Pattern configuration and randomization impact data quality and diversity.
- **Scalability:** Very large trajectories or datasets may require optimization for speed and memory.

---

## 9. Design Decisions
- **Pattern-Based Generation:** Each pattern has a dedicated method for clarity and extensibility.
- **Variance Injection:** Adds realistic unpredictability to timing and performance metrics.
- **Post-Generation Validation:** Ensures all trajectories are valid and metrics are up to date.

---

## 10. Outcome
This module enables the creation of diverse, realistic, and configurable synthetic agent trajectories, supporting robust model development and evaluation. 