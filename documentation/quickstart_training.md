# Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed on your system
2. **OpenAI API Key** for data generation
3. **Required packages** (see requirements.txt)

## Installation

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start: Generate Training Data

### Step 1: Generate Normal Traces
```bash
python generate_training_data.py --normal
```
This will create 50 normal agent interaction traces in `training_dataset/normal/`.

### Step 2: Generate Anomaly Traces
```bash
python generate_training_data.py --anomaly
```
This will create 2 traces for each of the 15 anomaly types in `training_dataset/anomaly/`.

### Step 3: Generate Both (Recommended)
```bash
python generate_training_data.py
```
This generates both normal and anomaly data in one command.

## Quick Start: Train Anomaly Detection Model

### Pipeline Overview

```mermaid
graph LR
    A[Generate Data] --> B[Train Models]
    B --> C[Save Artifacts]
    C --> D[Evaluate Performance]
    
    subgraph "Data Generation"
        A1[GPT-4o API] --> A2[Training Traces]
        A2 --> A3[Normal & Anomaly Data]
    end
    
    subgraph "Model Training"
        B1[Graph Processing] --> B2[DGI Encoder]
        B2 --> B3[Autoencoder]
        B3 --> B4[GMM Model]
    end
    
    subgraph "Outputs"
        C1[Model Files] --> C2[Visualizations]
        C2 --> C3[Performance Metrics]
    end
    
    A --> A1
    B --> B1
    C --> C1
```

### Step 1: Run Training Pipeline
```bash
python training_pipeline.py
```

This will:
- Load the generated training data
- Convert traces to graphs
- Extract motifs and features
- Train DGI encoder and autoencoder
- Train Gaussian Mixture Model for OOD detection
- Generate comprehensive visualizations
- Save trained models

### Step 2: Check Results
After training completes, check the outputs:

- **Models**: `training/model/`
- **Visualizations**: `training/visuals/`
- **Training logs**: Console output

## Expected Output

### Directory Structure
```
modules/
├── training_dataset/
│   ├── normal/          # Generated normal traces
│   └── anomaly/         # Generated anomaly traces
├── training/
│   ├── model/           # Trained models
│   ├── visuals/         # Generated plots
│   ├── checkpoints/     # Training checkpoints
│   └── cache/           # Cached computations
└── eval_dataset/        # Evaluation data (if using)
```

### Key Files Generated
- `training/model/autoencoder_final.pth` - Trained autoencoder
- `training/model/dgi_encoder_final.pth` - Trained DGI encoder
- `training/visuals/confusion_matrix.png` - Model performance
- `training/visuals/roc_curve.png` - ROC curve analysis

## Customization

### Adjust Data Generation
Edit `generate_training_data.py` to modify:
- Number of traces: `NUM_NORMAL`, `NUM_ANOMALY_PER_TYPE`
- Available tools/agents: `AVAILABLE_TOOLS`, `AVAILABLE_AGENTS`
- Anomaly types: `ANOMALY_TYPES`

### Adjust Training Parameters
Edit `training_pipeline.py` to modify:
- Training epochs: `EPOCHS_MOTIF`, `AUTOENCODER_EPOCHS`
- Batch size: `BATCH_SIZE_MOTIF`
- Learning rate: `LEARNING_RATE`
- Model architecture: Hidden dimensions, layers, etc.

## Common Use Cases

### 1. Quick Experiment
```bash
# Generate minimal dataset
python generate_training_data.py --normal
python generate_training_data.py --anomaly

# Train with default settings
python training_pipeline.py
```

### 2. Production Training
```bash
# Generate larger dataset
# Edit NUM_NORMAL = 200, NUM_ANOMALY_PER_TYPE = 5 in generate_training_data.py
python generate_training_data.py

# Train with caching enabled
python training_pipeline.py --cache-prefix production_v1
```

### 3. Research/Development
```bash
# Clear cache for fresh start
python training_pipeline.py --clear-cache

# Try different threshold methods
python training_pipeline.py --threshold-method percentile_95
python training_pipeline.py --threshold-method mean_plus_2std
```

## Troubleshooting

### Common Issues

1. **OpenAI API Error**
   ```
   ERROR: OpenAI API key not found!
   ```
   **Solution**: Set your API key: `export OPENAI_API_KEY="your-key"`

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce batch size in `training_pipeline.py`

3. **Cache Issues**
   ```
   Cache corruption detected
   ```
   **Solution**: Clear cache: `python training_pipeline.py --clear-cache`

4. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named 'torch_geometric'
   ```
   **Solution**: Install missing packages: `pip install torch-geometric`

### Performance Tips

1. **Use Caching**: Don't disable cache unless debugging
2. **GPU Usage**: The system automatically uses GPU if available
3. **Batch Size**: Adjust based on your hardware capabilities
4. **Data Size**: Start with smaller datasets for testing

## Next Steps

After successful training:

1. **Analyze Results**: Check visualizations in `training/visuals/`
2. **Evaluate Performance**: Review confusion matrix and ROC curves
3. **Customize Models**: Adjust architecture or training parameters
4. **Scale Up**: Increase dataset size for better performance
5. **Deploy**: Use trained models for real-time anomaly detection

## Support

- Check the main `README.md` for detailed documentation
- Review `TECHNICAL_DOCS.md` for implementation details
- Examine generated visualizations for model insights
- Monitor console output for training progress and errors

## Example Output

Successful training should show:
```
🚀 Starting Anomaly Detection Pipeline...
📱 Using device: cpu
⚙️  Cache setting: Enabled

📂 Loading trace data...
✅ Loaded 50 normal traces (missing metadata: 0)
✅ Loaded 30 anomaly traces (missing metadata: 0)

🔄 Splitting normal data into train/test...
✅ Normal train: 40, test: 10

🔄 Converting traces to graphs...
✅ Graphs: train 40, test 10, anomaly 30

🔄 Training DGI encoder...
✅ DGI encoder training completed and saved

🔄 Training autoencoder...
✅ Autoencoder training completed and saved

🎉 Pipeline completed successfully!
📊 Final Results:
   Accuracy: 0.9250
   Precision: 0.9000
   Recall: 0.9000
   F1 Score: 0.9000
   AUC-ROC: 0.9500
   AUC-PR: 0.9500
   Threshold: 0.123456
```

This indicates a well-performing anomaly detection model! 