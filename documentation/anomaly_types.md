# Standardized Anomaly Types System

This document describes the standardized anomaly types system for multi-agent trajectory analysis and how to use the LLM-powered update functionality.

## Overview

The system provides a curated list of 20 standardized anomaly types that can occur in multi-agent trajectories, organized into 7 thematic categories. Each anomaly type includes detailed descriptions, step-level characteristics, and detection methods.

## Standardized Anomaly Types

### Planning and Decomposition Anomalies
- **Planning Failure**: Errors in creating, updating, or managing plans
- **Decomposition Error**: Faulty breakdown of tasks into sub-tasks

### Execution and Tool-Related Anomalies
- **Tool Calling Error**: Incorrect or inappropriate tool invocation
- **Inadequate Validation of Tool Results**: Failure to verify tool outputs
- **Suboptimal Path**: Inefficient sequencing of steps or tools

### Collaboration and Handoff Anomalies
- **Agent Handoff Error**: Incorrect assignment to wrong sub-agent
- **Feedback Loop Failure**: Broken feedback mechanisms between agents

### Memory and State Anomalies
- **Memory Inconsistency**: Inconsistent recall/updating of shared memory
- **Error Propagation**: Errors carrying over without mitigation

### Task Completion and Output Anomalies
- **Task Not Completed**: Abrupt end without resolution
- **Partial or Incomplete Answer**: Fragmented output despite completion
- **Irrelevant or Off-Topic Answer**: Response deviates from query
- **Overconfidence in Incorrect Answer**: Asserting wrong info as correct

### Behavioral and Reasoning Anomalies
- **Loop or Repetition**: Repeated actions without progress
- **Misinterpretation of Question or Context**: Wrong understanding of input
- **Lack of Alternative Strategy**: No fallback on failure
- **Failure to Recover from Error**: Ignores errors without retry

### Specialized Anomalies
- **Hallucination**: Generating unsupported information
- **Handling of Fictional or Impossible Queries**: Poor response to nonsensical inputs

## Files

### Core Files
- `dataset_generator/anomaly_types.py`: Main module with standardized types and LLM updater
- `update_anomaly_types.py`: Command-line script for updating datasets
- `ANOMALY_TYPES_README.md`: This documentation file

### Updated Files
- `dataset_generator/generate_training_data.py`: Now imports standardized types
- `dataset_generator/generate_eval_data.py`: Now imports standardized types
- `training_pipeline.py`: Already supports anomaly types for visualization
- `eval_pipeline.py`: Already supports anomaly types for visualization
- `dynamic_pipline.py`: Already supports anomaly types for visualization

## Usage

### 1. View Standardized Anomaly Types

```bash
python3 update_anomaly_types.py --action summary
```

This displays all 20 standardized anomaly types with their descriptions.

### 2. Dry Run (Test Classification)

Before updating your datasets, you can test the LLM classification:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Test on a specific directory
python3 update_anomaly_types.py --action dry-run --directory training_dataset/anomaly/

# Test on all directories
python3 update_anomaly_types.py --action dry-run
```

This will show you what anomaly types the LLM would assign to your existing files without making any changes.

### 3. Update Datasets

To actually update your datasets with standardized anomaly types:

```bash
# Update all anomaly directories
python3 update_anomaly_types.py --action update

# Update specific directory
python3 update_anomaly_types.py --action update --directory eval_dataset/anomaly/

# Update without creating backups (not recommended)
python3 update_anomaly_types.py --action update --no-backup

# Use a different LLM model
python3 update_anomaly_types.py --action update --model gpt-3.5-turbo
```

### 4. Programmatic Usage

You can also use the system programmatically:

```python
from dataset_generator.anomaly_types import (
    get_anomaly_types, 
    get_anomaly_description, 
    AnomalyTypeUpdater
)

# Get all anomaly types
types = get_anomaly_types()

# Get description for specific type
desc = get_anomaly_description("Tool Calling Error")

# Initialize updater
updater = AnomalyTypeUpdater(api_key="your-key")

# Update a single file
success = updater.update_trace_file("path/to/trace.json")

# Update a directory
stats = updater.update_dataset_directory("path/to/anomaly/directory/")
```

## LLM Classification Process

The system uses GPT-4 (or another specified model) to automatically classify anomaly types by:

1. **Analyzing trace data**: Question, steps, metadata, errors
2. **Comparing against standardized types**: Using the curated list of 20 types
3. **Selecting 1-3 types**: Choosing the most relevant anomaly types
4. **Updating metadata**: Adding `anomaly_types` field to trace metadata

### Classification Prompt

The LLM receives a structured prompt that includes:
- Trace summary (question, completion status, errors)
- Step-by-step summary (first 10 steps)
- Complete list of standardized anomaly types with descriptions
- Instructions to select 1-3 most relevant types

### Output Format

The system expects the LLM to return a JSON array of anomaly type names:

```json
["Tool Calling Error", "Inadequate Validation of Tool Results"]
```

## Backup and Safety

- **Automatic backups**: By default, `.backup` files are created before any changes
- **Dry run mode**: Test classification without making changes
- **Error handling**: Failed updates are logged and don't affect other files
- **Validation**: Checks for valid JSON and proper anomaly type names

## Requirements

### Required Packages
```bash
pip install openai
```

### Optional Packages (for full functionality)
```bash
pip install plotly kaleido  # For Sankey diagrams in training pipeline
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Integration with Training Pipeline

The training pipeline automatically uses the standardized anomaly types for:

1. **Visualization**: Anomaly type distribution plots
2. **Analysis**: Detection accuracy by anomaly type
3. **Clustering**: Anomaly type distribution in clusters
4. **Reporting**: Detailed analysis of model performance per anomaly type

## Migration from Old System

If you have existing datasets with old anomaly type names, the LLM updater will:

1. **Analyze existing traces**: Understand the current anomaly patterns
2. **Map to new types**: Convert old names to standardized types
3. **Preserve metadata**: Keep all other trace information intact
4. **Create backups**: Save original files before changes

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: OpenAI API key not provided
   ```
   Solution: Set `OPENAI_API_KEY` environment variable or use `--api-key`

2. **OpenAI Package Not Installed**
   ```
   Error: OpenAI package not available
   ```
   Solution: `pip install openai`

3. **Directory Not Found**
   ```
   Directory training_dataset/anomaly/ not found
   ```
   Solution: Check your directory structure and paths

4. **Classification Failures**
   - Check API key validity
   - Verify internet connection
   - Try a different model (e.g., `gpt-3.5-turbo`)

### Getting Help

- Run with `--action summary` to see all available anomaly types
- Use `--action dry-run` to test before making changes
- Check backup files (`.backup` extension) if you need to revert changes
- Review the classification output for unexpected results

## Contributing

To add new anomaly types or modify existing ones:

1. Edit `dataset_generator/anomaly_types.py`
2. Add the new type to `ANOMALY_TYPES` list
3. Add description to `ANOMALY_DESCRIPTIONS`
4. Update `ANOMALY_CATEGORIES` if needed
5. Test with `--action dry-run`

## License

This system is part of the multi-agent trajectory analysis project. Please refer to the main project license for usage terms. 