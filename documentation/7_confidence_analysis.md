# 7. Confidence Analysis Module (confidence_analysis.py)

## Overview
This module provides comprehensive analysis of model confidence for anomaly detection in agent trajectory data. It supports confidence distribution plots, calibration analysis, and confidence-based error analysis for multiple models.

---

## 1. Class and Function Reference

### `ConfidenceAnalyzer`
- **Purpose:** Main class for analyzing model confidence, calibration, and error types.
- **Key Methods:**
  - `__init__(config: Dict, output_dir: str = "charts")`
  - `plot_confidence_distributions(model_results, test_df, test_features) -> str`
  - `plot_confidence_vs_performance(model_results, test_df, test_features) -> str`
  - `plot_calibration_analysis(model_results, test_df, test_features) -> str`
  - `plot_confidence_error_analysis(model_results, test_df, test_features) -> str`
  - `generate_comprehensive_confidence_report(model_results, test_df, test_features) -> str`
  - `_get_confidence_scores(model_data, test_features, model_name) -> Optional[np.ndarray]`
  - `_normalize_scores_to_probabilities(scores: np.ndarray) -> np.ndarray`

### Utility Functions
- `setup_matplotlib_for_plotting()`: Configures plotting style and backend.
- Uses `ensure_directory` from utils for output management.

---

## 2. Configuration Parameters

| Parameter                | Type    | Description                                              | Default      |
|--------------------------|---------|----------------------------------------------------------|--------------|
| `output_dir`             | str     | Directory to save output charts                          | "charts"    |
| `visualization`          | dict    | Visualization config (figure size, style, etc.)          | `{}`         |
| `figure_settings.dpi`    | int     | Dots per inch for saved figures                          | 300          |
| `figure_settings.format` | str     | Output image format (e.g., 'png', 'pdf')                 | 'png'        |
| `figure_settings.bbox_inches` | str | Bounding box for saving figures                          | 'tight'      |
| `model_colors`           | dict    | Color mapping for each model type                        | (preset)     |

**Example Config:**
```python
config = {
    'visualization': {
        'figure_settings': {
            'dpi': 300,
            'format': 'png',
            'bbox_inches': 'tight'
        }
    }
}
```

---

## 3. Workflow Pseudocode

```python
# Pseudocode for comprehensive confidence analysis
analyzer = ConfidenceAnalyzer(config)

# 1. Plot confidence distributions
analyzer.plot_confidence_distributions(model_results, test_df, test_features)

# 2. Plot confidence vs performance
analyzer.plot_confidence_vs_performance(model_results, test_df, test_features)

# 3. Plot calibration analysis
analyzer.plot_calibration_analysis(model_results, test_df, test_features)

# 4. Plot confidence-based error analysis
analyzer.plot_confidence_error_analysis(model_results, test_df, test_features)

# 5. Generate comprehensive report
analyzer.generate_comprehensive_confidence_report(model_results, test_df, test_features)
```

---

## 4. Sample Output

**Example: Confidence Distribution Output**
- File: `charts/confidence_distributions.png`
- Structure: Multi-panel plot with histograms and KDEs for each model, showing normal vs. anomalous confidence scores, annotated with mean and std.

**Example: Calibration Output**
- File: `charts/calibration_analysis.png`
- Structure: Calibration curve for each model, with Brier score annotation.

**Example: Comprehensive Report**
- File: `charts/comprehensive_confidence_analysis.png`
- Structure: 8-panel figure (2 models × 4 analyses: distribution, performance, calibration, error).

---

## 5. Data Structures and Integration Points

- **Input:**
  - `model_results`: Dict[str, Any] (model name → result dict with model, method, scores, etc.)
  - `test_df`: pd.DataFrame (must include `is_anomalous` column)
  - `test_features`: np.ndarray (feature matrix)
- **Output:**
  - Saved figures (PNG/PDF)
  - Paths to output files (for downstream reporting/presentation)
- **Integration:**
  - Consumes model outputs from the models module
  - Used in reporting, presentation, and debugging

---

## 6. Edge Cases and Error Handling
- Handles missing models, errors in model results, and plotting exceptions
- Skips models with errors and logs warnings
- Robust to missing or malformed data (e.g., missing scaler or PCA for SVM)
- Fallbacks for normalization (sigmoid or min-max)
- Hides unused subplots if fewer models than expected

---

## 7. Limitations and Concerns
- Confidence scores may not be true probabilities; calibration is essential
- Model output format must be consistent; custom models may require adaptation
- Designed for moderate numbers of models; very large ensembles may need layout changes

---

## 8. Design Decisions
- Multi-panel layout for holistic view
- Publication-quality plots with consistent styling
- Modular, extensible class structure

---

## 9. Outcome
This module enables detailed, interpretable, and visually rich analysis of model confidence, supporting both model debugging and effective communication of results. 