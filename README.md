# DGI Anomaly Detection

A deep learning-based anomaly detection system using Deep Graph Infomax (DGI) and Autoencoder for detecting anomalies in agent conversation traces.

## Usage

```bash
python main.py [dataset_name] [run_number] [additional_datasets...]
```

## Features

- Graph-based anomaly detection using DGI
- Autoencoder for reconstruction-based scoring
- Support for multiple scoring methods (AE loss, Mahalanobis distance)
- Configurable model parameters and training settings 