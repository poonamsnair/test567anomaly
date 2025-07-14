# Anomaly Detection Pipeline

## Overview

This repository provides a complete pipeline for anomaly detection in multi-agent reasoning traces. It covers data generation, model training, and evaluation, using a hybrid approach that combines graph neural networks, autoencoders, and out-of-distribution (OOD) detection.

## Main Components

1. **Training Data Generation**  
   - Generates synthetic traces (normal and anomalous) for model training.

2. **Training Pipeline**  
   - Trains models using graph-based and feature-based methods.

3. **Evaluation Data Generation**  
   - Produces domain-specific evaluation data (e.g., for financial services).

4. **Evaluation Pipeline**  
   - Evaluates trained models and provides performance metrics.

## How It Works

- **Data Generation:**  
  Synthetic traces are created for both normal and anomalous agent interactions.

- **Training:**  
  The pipeline processes traces into graphs, extracts features, and trains models (graph encoder, autoencoder, GMM).

- **Evaluation:**  
  New data is processed and scored using the trained models, combining reconstruction error and OOD detection for robust anomaly identification.

## Usage

**Generate Training Data:**
```bash
python generate_training_data.py
```

**Train the Model:**
```bash
python training_pipeline.py
```

**Generate Evaluation Data:**
```bash
python generate_eval_data.py --categories ALL --num_traces 15
```

**Evaluate the Model:**
```bash
python eval_pipeline.py
```

## Output

- Trained models and artifacts are saved in the `training/` directory.
- Evaluation results and metrics are saved in the `eval/` directory.

## Key Features

- Hybrid anomaly detection (reconstruction + OOD)
- Graph neural networks for motif learning
- Sentence transformer embeddings for semantic features
- Detailed evaluation metrics and visualizations

## Requirements

- torch, torch-geometric, networkx, numpy, scikit-learn, sentence-transformers, matplotlib, seaborn, tqdm 