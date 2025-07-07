# Module: evaluation.py

## Overview
This module implements a comprehensive evaluation framework for unsupervised anomaly detection in agent trajectory data. It ensures proper data splitting, threshold calibration, metric calculation, and reporting for robust model assessment.

---

## 1. Class and Function Reference

### `AnomalyDetectionEvaluator`
- **Purpose:** Main class for splitting data, calibrating thresholds, evaluating models, and generating reports.
- **Key Methods:**
  - `__init__(config: Dict)`
  - `create_unsupervised_data_splits(features_df, random_state=42)`
  - `calibrate_threshold(model_results, val_df, val_features)`
  - `evaluate_model(model_results, test_df, test_features, threshold)`
  - `evaluate_by_anomaly_type(model_results, test_df, test_features, threshold)`
  - `generate_evaluation_report(results)`
  - `export_results(results, filepath)`
- **Internal Methods:**
  - `_calculate_overall_detection_recall`, `_calculate_false_discovery_rate`, `_calculate_severity_weighted_performance`

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                                      |
|--------------------------|---------|------------------------------------------------------------------|
| `data_split`             | dict    | Train/validation/test ratios                                     |
| `threshold_calibration`  | list    | List of calibration methods (roc_optimization, pr_optimization, f1_maximization, fixed_percentile_95, knee_point_detection) |
| `metrics`                | list    | List of metrics to calculate (precision, recall, f1, auc_roc, auc_pr, etc.) |

**Example config:**
```yaml
evaluation:
  data_split:
    train_ratio: 0.6
    validation_ratio: 0.2
    test_ratio: 0.2
  threshold_calibration:
    - roc_optimization
    - pr_optimization
    - f1_maximization
    - fixed_percentile_95
    - knee_point_detection
  metrics:
    - precision
    - recall
    - f1
    - accuracy
    - auc_roc
    - auc_pr
    - silhouette_score
    - calinski_harabasz_score
    - davies_bouldin_score
    - overall_detection_recall
    - false_discovery_rate
    - severity_weighted_performance
```

---

## 3. Workflow Pseudocode

```python
# Split data
train_df, val_df, test_df = evaluator.create_unsupervised_data_splits(features_df)

# Calibrate threshold
thresholds = evaluator.calibrate_threshold(model_results, val_df, val_features)

# Evaluate model
metrics = evaluator.evaluate_model(model_results, test_df, test_features, threshold)

# Evaluate by anomaly type
type_metrics = evaluator.evaluate_by_anomaly_type(model_results, test_df, test_features, threshold)

# Generate report
report = evaluator.generate_evaluation_report({
    'model_results': model_results,
    'threshold_results': thresholds,
    'type_analysis': type_metrics,
    'total_test_samples': len(test_df),
    'anomaly_rate': test_df['is_anomalous'].mean() if 'is_anomalous' in test_df else 0.0
})

# Export results
report_path = 'results/evaluation_report.json'
evaluator.export_results(report, report_path)
```

- **Data splitting** ensures only normal data is used for training.
- **Threshold calibration** supports multiple methods for robust detection.
- **Metrics** include both supervised and unsupervised measures.
- **Reporting** provides summaries, comparisons, and actionable recommendations.

---

## 4. Sample Output (Evaluation Report)

```json
{
  "summary": {
    "best_model": "ensemble",
    "best_f1_score": 0.91,
    "models_evaluated": 3,
    "total_test_samples": 200,
    "anomaly_rate": 0.25
  },
  "model_comparison": {
    "isolation_forest": {"f1": 0.88, "precision": 0.85, "recall": 0.92, "auc_roc": 0.93, "auc_pr": 0.89},
    "one_class_svm": {"f1": 0.81, "precision": 0.78, "recall": 0.85, "auc_roc": 0.87, "auc_pr": 0.82},
    "ensemble": {"f1": 0.91, "precision": 0.89, "recall": 0.93, "auc_roc": 0.95, "auc_pr": 0.92}
  },
  "threshold_analysis": {
    "ensemble:roc_optimization": {"f1_score": 0.91, "precision": 0.89, "recall": 0.93},
    ...
  },
  "anomaly_type_analysis": {
    "INFINITE_LOOPS": {"count": 20, "detection_rate": 0.95, "recall": 0.95, "precision": 0.90},
    "normal": {"count": 150, "false_positive_rate": 0.05, "specificity": 0.95},
    ...
  },
  "recommendations": [
    "Use ensemble model for deployment (best F1: 0.91)",
    "one_class_svm: Low precision (0.78) - many false positives"
  ]
}
```

---

## 5. Data Structures and Metrics

### Data Splits
- `train_df`, `val_df`, `test_df`: pandas DataFrames with features and labels

### Thresholds
- Dict of calibration methods to threshold values

### Metrics
- `precision`, `recall`, `f1`, `accuracy`, `auc_roc`, `auc_pr`, `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `overall_detection_recall`, `false_discovery_rate`, `severity_weighted_performance`, etc.

### Report Structure
- `summary`, `model_comparison`, `threshold_analysis`, `anomaly_type_analysis`, `recommendations`

---

## 6. Edge Cases and Error Handling
- **All normal or all anomalous data:** Handles splits and metrics gracefully.
- **Duplicate columns:** Removes duplicates before splitting.
- **Small/imbalanced datasets:** Warns and adjusts splits/metrics as needed.
- **Metric calculation errors:** Catches/logs exceptions, returns defaults.
- **Export:** Converts numpy arrays to lists for JSON serialization.

---

## 7. Integration Points
- **Input:** Takes features, model results, and predictions from previous modules.
- **Output:** Returns metrics, reports, and exported JSON files.
- **Downstream:** Used by confidence analysis, synthetic data analysis, and presentation modules.
- **Config:** Reads from a config dict (YAML/JSON supported via utility functions).

---

## 8. Limitations and Concerns
- **Label Dependency for Evaluation:** Requires labeled data for validation and test sets, even though training is unsupervised.
- **Threshold Sensitivity:** Model performance is highly sensitive to threshold calibration; poor calibration can degrade results.
- **Metric Selection:** Some metrics may not be meaningful for highly imbalanced or small datasets.
- **Interpretability:** Some unsupervised metrics (e.g., silhouette score) may be hard to interpret in anomaly detection context.

---

## 9. Design Decisions
- **Strict Data Splitting:** Enforces unsupervised learning principles by never using anomaly labels during training.
- **Comprehensive Reporting:** Provides detailed breakdowns and recommendations for model selection and deployment.

---

## 10. Outcome
This module provides a robust, extensible, and transparent evaluation pipeline for unsupervised anomaly detection, supporting both model development and deployment decisions. 