# Module: anomaly_injection.py

## Overview
This module implements a configurable anomaly injection system for agent trajectory data, supporting the creation of realistic anomalous patterns for model training and evaluation. It covers 10 specific anomaly types, each with configurable severity.

---

## 1. Class and Function Reference

### `AnomalyInjector`
- **Purpose:** Main class for injecting anomalies into agent trajectories, supporting 10 anomaly types and multiple severity levels.
- **Key Methods:**
  - `__init__(config: Dict, random_seed: int = 42)`
  - `inject_anomalies(normal_trajectories: List[AgentTrajectory], num_anomalous: int) -> List[AgentTrajectory]`
  - `get_anomaly_statistics(anomalous_trajectories: List[AgentTrajectory]) -> Dict`
  - `_select_anomaly_type() -> AnomalyType`
  - `_create_anomalous_trajectory(base_trajectory: AgentTrajectory, anomaly_type: AnomalyType) -> AgentTrajectory`
  - `_ensure_connectivity(trajectory: AgentTrajectory) -> None`
  - `_insert_anomalous_node_connected(trajectory, new_node)`
  - `_inject_*` methods for each anomaly type (see below)

#### Anomaly Injection Methods
- `_inject_infinite_loops`, `_inject_suboptimal_paths`, `_inject_tool_failure_cascades`, `_inject_planning_paralysis`, `_inject_memory_inconsistencies`, `_inject_timeout_cascades`, `_inject_handoff_failures`, `_inject_validation_loops`, `_inject_context_drift`, `_inject_incomplete_responses`
  - Each method modifies nodes, edges, and metadata to simulate a specific anomaly type and severity.

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                                      |
|--------------------------|---------|------------------------------------------------------------------|
| `random_seed`            | int     | Seed for reproducibility                                         |
| `anomaly_types`          | dict    | Dict of anomaly type configs (ratio, severity, etc.)             |
| `severity_levels`        | dict    | Dict of severity configs (degradation ranges, etc.)              |
| `ratio`                  | float   | Probability ratio for each anomaly type                          |
| `severity`               | str     | Severity level for each anomaly type (LOW, MEDIUM, HIGH, CRITICAL) |
| `degradation_range`      | tuple   | Range of completion rate degradation for each severity           |

**Example config:**
```yaml
random_seed: 42
anomaly_injection:
  anomaly_types:
    INFINITE_LOOPS: {ratio: 0.1, severity: 'MEDIUM'}
    SUBOPTIMAL_PATHS: {ratio: 0.1, severity: 'LOW'}
    TOOL_FAILURE_CASCADES: {ratio: 0.1, severity: 'HIGH'}
    PLANNING_PARALYSIS: {ratio: 0.1, severity: 'LOW'}
    MEMORY_INCONSISTENCIES: {ratio: 0.1, severity: 'MEDIUM'}
    TIMEOUT_CASCADES: {ratio: 0.1, severity: 'HIGH'}
    HANDOFF_FAILURES: {ratio: 0.1, severity: 'MEDIUM'}
    VALIDATION_LOOPS: {ratio: 0.1, severity: 'LOW'}
    CONTEXT_DRIFT: {ratio: 0.1, severity: 'MEDIUM'}
    INCOMPLETE_RESPONSES: {ratio: 0.1, severity: 'CRITICAL'}
  severity_levels:
    LOW: {degradation_range: [0.01, 0.05]}
    MEDIUM: {degradation_range: [0.05, 0.15]}
    HIGH: {degradation_range: [0.15, 0.3]}
    CRITICAL: {degradation_range: [0.3, 0.5]}
```

---

## 3. Workflow Pseudocode

```python
# Main anomaly injection loop
for i in range(num_anomalous):
    base_traj = random.choice(normal_trajectories)
    anomaly_type = select_anomaly_type_by_ratio(config['anomaly_types'])
    anomalous_traj = create_anomalous_trajectory(base_traj, anomaly_type)
    anomalous_trajectories.append(anomalous_traj)

# For each anomaly type, call the corresponding _inject_* method
# Each method modifies nodes, edges, and metadata to simulate the anomaly
```

- **Severity levels** control the subtlety and impact of the injected anomaly (e.g., number of loop iterations, failure count, completion rate degradation).
- **Post-processing** ensures all nodes are connected and metrics are recalculated.

---

## 4. Sample Output (Anomalous Trajectory JSON)

```json
{
  "trajectory_id": "uuid2",
  "is_anomalous": true,
  "anomaly_types": ["INFINITE_LOOPS"],
  "anomaly_severity": "MEDIUM",
  "nodes": [
    {"node_id": "uuid1", "node_type": "USER_ENQUIRY", ...},
    {"node_id": "uuid2", "node_type": "PLANNING", ...},
    {"node_id": "uuid3", "node_type": "REASONING", "is_anomalous": true, "anomaly_type": "INFINITE_LOOPS", ...},
    ...
  ],
  "edges": [
    {"source_node_id": "uuid1", "target_node_id": "uuid2", "edge_type": "execution_flow", ...},
    ...
  ],
  "completion_rate": 0.7,
  "status": "incomplete",
  "anomaly_metadata": {"loop_iteration": 2, "loop_size": 3, ...}
}
```

---

## 5. Anomaly Types and Data Structures

### Anomaly Types (`AnomalyType`)
- `INFINITE_LOOPS`, `SUBOPTIMAL_PATHS`, `TOOL_FAILURE_CASCADES`, `PLANNING_PARALYSIS`, `MEMORY_INCONSISTENCIES`, `TIMEOUT_CASCADES`, `HANDOFF_FAILURES`, `VALIDATION_LOOPS`, `CONTEXT_DRIFT`, `INCOMPLETE_RESPONSES`

### Severity Levels (`SeverityLevel`)
- `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

### Node/Edge Anomaly Attributes
- `is_anomalous`, `anomaly_type`, `anomaly_severity`, `anomaly_metadata`, `status`, `error_code`, `error_message`, `retry_count`, etc.

---

## 6. Edge Cases and Error Handling
- **Floating/disconnected nodes:** `_ensure_connectivity` connects all nodes post-injection; warnings are logged.
- **Invalid severity:** Falls back to `MEDIUM` if severity string is invalid.
- **Degenerate patterns:** If not enough nodes for a pattern, injection is skipped or minimal.
- **Randomization:** All random choices are seeded for reproducibility.

---

## 7. Integration Points
- **Input:** Takes a list of normal `AgentTrajectory` objects (from data generation).
- **Output:** Returns a list of anomalous `AgentTrajectory` objects (can be serialized to JSON/pickle).
- **Downstream:** Used by graph processing, feature engineering, and evaluation modules.
- **Config:** Reads from a config dict (YAML/JSON supported via utility functions).

---

## 8. Limitations and Concerns
- **Synthetic Nature:** Injected anomalies may not capture all real-world failure patterns.
- **Parameter Sensitivity:** The impact of anomalies depends on configuration; poor settings may yield unrealistic data.
- **Complexity:** Some anomaly types (e.g., context drift) require careful design to avoid trivial detection.

---

## 9. Design Decisions
- **Dedicated Methods:** Each anomaly type has a dedicated method for clarity and extensibility.
- **Severity Control:** Severity levels allow for nuanced control over anomaly impact.
- **Post-Injection Validation:** Ensures all trajectories remain connected and metrics are recalculated.

---

## 10. Outcome
This module enables the creation of diverse, realistic, and configurable anomalous data for robust model training and evaluation, supporting both research and practical deployment scenarios. 