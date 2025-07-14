import os
import json
import glob
import random
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sentence_transformers import SentenceTransformer
from itertools import combinations
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.mixture import GaussianMixture  # <-- Add this import
import re
from collections import Counter

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.covariance._robust_covariance')

# === CONFIGURATION ===
DIG_ROOT = 'training'
MODEL_DIR = os.path.join(DIG_ROOT, 'model')
CHECKPOINT_DIR = os.path.join(DIG_ROOT, 'checkpoints')
VISUALS_DIR = os.path.join(DIG_ROOT, 'visuals')
CACHE_DIR = os.path.join(DIG_ROOT, 'cache')
EVAL_DIR = 'eval'  # New evaluation results directory
NORMAL_DIR_NEW = 'eval_dataset/normal/'
ANOMALY_DIR_NEW = 'eval_dataset/anomaly/'
BATCH_SIZE_MOTIF = 64
EPOCHS_MOTIF = 300
AUTOENCODER_EPOCHS = 500
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 15
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)  # Create eval directory

# === UTILITY FUNCTIONS ===
motif_embed_cache = {}

def motif_hash(subg):
    edges = sorted([tuple(sorted(e)) for e in subg.edges])
    node_texts = tuple(sorted([subg.nodes[n]['text'] for n in subg.nodes]))
    return hash((tuple(edges), node_texts))

def motif_to_pyg_cached(subg, embedder, device='cpu'):
    h = motif_hash(subg)
    if h in motif_embed_cache:
        data = motif_embed_cache[h]
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        return data
    node_texts = [subg.nodes[n]['text'] for n in subg.nodes]
    node_embeds = embedder.encode(node_texts, show_progress_bar=False)
    x = torch.tensor(node_embeds, dtype=torch.float, device=device)
    edge_index = torch.tensor(np.array(list(subg.edges)).T if subg.edges else np.zeros((2,0), dtype=int), dtype=torch.long, device=device)
    data = Data(x=x, edge_index=edge_index)
    motif_embed_cache[h] = data
    return data

def validate_trace_metadata(trace, missing_metadata_counter=None):
    meta = trace.get('metadata', {})
    required_fields = ['num_steps', 'task_completed', 'tools_called', 'agents_called', 'errors', 'duration']
    missing_fields = [field for field in required_fields if field not in meta]
    if len(missing_fields) > len(required_fields) // 2:
        if missing_metadata_counter is not None:
            missing_metadata_counter[0] += 1
        return None
    if missing_fields:
        if missing_metadata_counter is not None:
            missing_metadata_counter[0] += 1
        meta['num_steps'] = meta.get('num_steps', len(trace.get('steps', [])))
        meta['task_completed'] = meta.get('task_completed', False)
        meta['tools_called'] = meta.get('tools_called', [])
        meta['agents_called'] = meta.get('agents_called', [])
        meta['errors'] = meta.get('errors', [])
        meta['duration'] = meta.get('duration', 'unknown')
    return trace

def load_json_files(directory, missing_metadata_counter=None):
    files = glob.glob(os.path.join(directory, '*.json'))
    data = []
    for f in files:
        with open(f, 'r') as fp:
            trace = json.load(fp)
            trace = validate_trace_metadata(trace, missing_metadata_counter)
            if trace is not None:
                data.append(trace)
    return data

def extract_trace_features(trace, embedder=None):
    meta = trace.get('metadata', {})
    question = trace.get('question', '')
    steps = trace.get('steps', [])
    step_texts = [s.get('content', '') for s in steps]
    agents = [s.get('agent', '') for s in steps]
    roles = [s.get('role', '') for s in steps]
    types = [s.get('type', '') for s in steps]
    num_steps = meta.get('num_steps', len(steps))
    tools_called = meta.get('tools_called', []) if isinstance(meta.get('tools_called'), list) else []
    agents_called = meta.get('agents_called', []) if isinstance(meta.get('agents_called'), list) else []
    errors = meta.get('errors', []) if isinstance(meta.get('errors'), list) else []
    duration = meta.get('duration', '0 minutes')
    duration_minutes = 0
    if isinstance(duration, str):
        try:
            num, unit = duration.split()
            num = float(num)
            duration_minutes = num if 'minute' in unit else num / 60 if 'second' in unit else 0
        except:
            pass
    if embedder and step_texts:
        content_embeds = embedder.encode(step_texts, show_progress_bar=False)
        content_embedding_mean = np.mean(content_embeds, axis=0).tolist()
        content_embedding_std = np.std(content_embeds, axis=0).mean()
    else:
        content_embedding_mean = [0.0] * 384
        content_embedding_std = 0.0
    # Improved keyword detection: word-based, case-insensitive, punctuation-stripped
    anomaly_keywords = ['error', 'timeout', 'failed', 'exception', 'invalid', 'missing', 'undefined']
    keyword_features = {}
    if step_texts:
        combined_text = ' '.join(step_texts).lower()
        combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
        words = combined_text.split()
        word_counts = Counter(words)
        for keyword in anomaly_keywords:
            keyword_features[f'contains_{keyword}'] = keyword in word_counts
    else:
        for keyword in anomaly_keywords:
            keyword_features[f'contains_{keyword}'] = False
    features = {
        'num_steps': num_steps,
        'duration_minutes': duration_minutes,
        'num_tools': len(tools_called),
        'num_agents': len(agents_called),
        'avg_step_length': np.mean([len(text) for text in step_texts]) if step_texts else 0,
        'max_step_length': max([len(text) for text in step_texts], default=0),
        'min_step_length': min([len(text) for text in step_texts], default=0),
        'step_length_std': np.std([len(text) for text in step_texts]) if len(step_texts) > 1 else 0,
        'unique_agents': len(set(agents)),
        'agent_entropy': -sum([(agents.count(a)/len(agents))*np.log2(agents.count(a)/len(agents)) for a in set(agents)]) if agents else 0,
        'tool_entropy': -sum([(tools_called.count(t)/len(tools_called))*np.log2(tools_called.count(t)/len(tools_called)) for t in set(tools_called)]) if tools_called else 0,
        'tool_usage_frequency': len(tools_called) / max(num_steps, 1),
        'steps_per_minute': num_steps / max(duration_minutes, 1),
        'question_length': len(question),
        'step_complexity': np.mean([len(text.split()) for text in step_texts]) if step_texts else 0,
        'repetitive_actions': len([a for a in agents if agents.count(a) > 2]),
        'tool_diversity': len(set(tools_called)) / max(len(tools_called), 1),
        'step_consistency': np.std([len(text) for text in step_texts]) if len(step_texts) > 1 else 0,
        'agent_switching_frequency': len([i for i in range(1, len(agents)) if agents[i] != agents[i-1]]),
        'tool_usage_pattern': hash(tuple(sorted(tools_called))) % 1000,
        'question_complexity': len(question.split()) / max(len(question), 1),
        'has_long_steps': any(len(text) > 500 for text in step_texts),
        'has_short_steps': any(len(text) < 10 for text in step_texts),
        'role_entropy': -sum([(roles.count(r)/len(roles))*np.log2(roles.count(r)/len(roles)) for r in set(roles)]) if roles else 0,
        'type_entropy': -sum([(types.count(t)/len(types))*np.log2(types.count(t)/len(types)) for t in set(types)]) if types else 0,
        'consecutive_repeat_count': sum(1 for i in range(1, len(step_texts)) if step_texts[i] == step_texts[i-1]),
        'tool_agent_overlap': len(set(tools_called) & set(agents_called)) / max(len(tools_called), 1),
        'avg_step_duration': duration_minutes / max(num_steps, 1),
        'question_tool_match': any(any(word in tool.lower() for word in question.lower().split()) for tool in tools_called),
        'content_embedding_std': content_embedding_std,
        'content_embedding_mean': content_embedding_mean,
        'contains_error': keyword_features.get('contains_error', False),
        'contains_timeout': keyword_features.get('contains_timeout', False),
        'contains_failed': keyword_features.get('contains_failed', False),
        'contains_exception': keyword_features.get('contains_exception', False),
    }
    max_streak = current_streak = 1
    for i in range(1, len(agents)):
        if agents[i] == agents[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    features['max_agent_streak'] = max_streak
    features['tools_called'] = tools_called
    features['agents_called'] = agents_called
    features['errors'] = errors
    return features

def traces_to_graphs(traces, label):
    graphs = []
    for trace in traces:
        features = extract_trace_features(trace)
        num_nodes = len(trace.get('steps', []))
        if num_nodes == 0:
            continue
        G = nx.DiGraph()
        G.graph['num_steps'] = features.get('num_steps', 0)
        G.graph['tools_called'] = len(features.get('tools_called', []))
        G.graph['agents_called'] = len(features.get('agents_called', []))
        G.graph['duration_minutes'] = features.get('duration_minutes', 0.0)
        G.graph['question'] = trace.get('question', '')
        G.graph['anomaly_types'] = trace.get('metadata', {}).get('anomaly_types', 'normal' if label == 0 else 'unknown')
        for i, step in enumerate(trace.get('steps', [])):
            G.add_node(i, text=step.get('content', ''), agent=step.get('agent', ''), role=step.get('role', ''), type=step.get('type', ''))
        for i in range(num_nodes-1):
            G.add_edge(i, i+1)
        graphs.append({'graph': G, 'label': label})
    return graphs

def extract_motifs_multi_size(G, motif_sizes=(2, 3, 4)):
    motifs = []
    seen = set()
    nodes = list(G.nodes)
    for motif_size in motif_sizes:
        for combo in combinations(nodes, motif_size):
            subg = G.subgraph(combo)
            if nx.is_connected(subg.to_undirected()):
                mapping = {n: i for i, n in enumerate(subg.nodes())}
                subg = nx.relabel_nodes(subg, mapping)
                h = motif_hash(subg)
                if h not in seen:
                    motifs.append(subg)
                    seen.add(h)
    return motifs

def extract_motifs_for_graphs(graphs, motif_sizes=(3, 4)):
    motifs_list = [extract_motifs_multi_size(item['graph'], motif_sizes) for item in tqdm(graphs, desc='Extracting motifs')]
    return motifs_list

def aggregate_embeddings(encoder, motifs_list, embedder, device='cpu'):
    encoder.eval()
    all_embeds = []
    with torch.no_grad():
        for motifs in tqdm(motifs_list, desc='Aggregating embeddings'):
            if not motifs:
                all_embeds.append(np.zeros(encoder.proj[-1].out_features))
                continue
            motif_embeds = [encoder.project(encoder(motif_to_pyg_cached(motif, embedder, device).x, motif_to_pyg_cached(motif, embedder, device).edge_index, torch.zeros(motif_to_pyg_cached(motif, embedder, device).x.size(0), dtype=torch.long, device=device))).cpu().numpy().squeeze() for motif in motifs]
            graph_embed = np.mean(motif_embeds, axis=0)
            all_embeds.append(graph_embed)
    return np.stack(all_embeds)

def remove_low_variance_features(X, threshold=1e-5, keep=None):
    if keep is None:
        variances = np.var(X, axis=0)
        keep = variances > threshold
        if np.sum(keep) == 0:
            keep[np.argmax(variances)] = True
    return X[:, keep], keep

def extract_feature_matrix(graphs, embedder=None):
    feature_list = []
    for item in graphs:
        trace = {'metadata': item['graph'].graph, 'steps': [item['graph'].nodes[n] for n in item['graph'].nodes], 'question': item['graph'].graph.get('question', '')}
        features = extract_trace_features(trace, embedder)
        scalar_features = [v for k, v in features.items() if k != 'content_embedding_mean' and isinstance(v, (int, float, bool))]
        vector_features = features.get('content_embedding_mean', [0.0] * 384)
        numeric_features = scalar_features + vector_features
        feature_list.append(numeric_features)
    return np.array(feature_list, dtype=np.float32)

def check_and_normalize_features(X, scaler=None):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler

# === CACHING UTILITIES ===
def load_cache(filename):
    """Load cached data if exists."""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def load_model_artifact(filename):
    """Load model artifacts from model directory."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def load_model_config():
    """Load model configuration from JSON file."""
    config_path = os.path.join(MODEL_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# === MODEL CLASSES ===
class DGIEncoder(nn.Module):
    """DGI Encoder for motif embeddings."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2, dropout=0.1):
        super(DGIEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)] + 
                                   [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] + 
                                   [GCNConv(hidden_dim, output_dim)])
        self.proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.relu(F.dropout(conv(x, edge_index), p=self.dropout, training=self.training))
        x = self.convs[-1](x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
    
    def project(self, x):
        return self.proj(x)
    
    def discriminate(self, x):
        return self.discriminator(x)

class ImprovedAutoencoder(nn.Module):
    """Improved Autoencoder for anomaly detection."""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.3, denoising=False):
        super(ImprovedAutoencoder, self).__init__()
        self.denoising = denoising
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        if self.denoising and self.training:
            x = x + torch.randn_like(x) * 0.05
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# === ANOMALY SCORING AND EVALUATION ===
def compute_anomaly_scores(autoencoder, gmm, data, device, ood_weight=0.4):
    autoencoder.eval()
    scores = []
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        for i in range(0, len(data_tensor), BATCH_SIZE_MOTIF):
            batch = data_tensor[i:i+BATCH_SIZE_MOTIF]
            decoded, encoded = autoencoder(batch)
            reconstruction_error = F.mse_loss(decoded, batch, reduction='none').mean(dim=1)
            cosine_error = 1 - F.cosine_similarity(decoded, batch, dim=1)
            l1_error = F.l1_loss(decoded, batch, reduction='none').mean(dim=1)
            recon_score = 0.5 * reconstruction_error + 0.3 * cosine_error + 0.2 * l1_error
            
            # OOD: Negative log-likelihood from GMM (higher = more OOD)
            batch_latents = encoded.cpu().numpy()
            ood_scores = -gmm.score_samples(batch_latents)  # Normalize if needed
            
            # Combined: Weighted sum (adjust ood_weight)
            combined_score = (1 - ood_weight) * recon_score.cpu().numpy() + ood_weight * ood_scores
            scores.extend(combined_score)
            
    return np.array(scores)

def evaluate_anomaly_detection(y_true, y_scores):
    """Evaluate anomaly detection with multiple metrics, finding optimal threshold."""
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
    best_f1 = 0
    best_threshold = thresholds[0]
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
    y_pred = (y_scores >= best_threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_pr': average_precision_score(y_true, y_scores),
        'auc_roc': roc_auc_score(y_true, y_scores),
        'threshold': best_threshold
    }

# === MAIN EVALUATION ===
def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load embedder
    try:
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Loaded SentenceTransformer successfully")
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        print("Using fallback embedder")
        class SimpleEmbedder:
            def __init__(self):
                self.dim = 384
            def encode(self, texts, show_progress_bar=False):
                import hashlib
                embeddings = []
                for text in texts:
                    hash_obj = hashlib.md5(text.encode())
                    hash_bytes = hash_obj.digest()
                    embedding = []
                    for i in range(0, len(hash_bytes), 4):
                        chunk = hash_bytes[i:i+4]
                        while len(chunk) < 4:
                            chunk += b'\x00'
                        val = int.from_bytes(chunk, 'big') / (2**32 - 1)
                        embedding.append(val)
                    while len(embedding) < 384:
                        embedding.extend(embedding[:min(384-len(embedding), len(embedding))])
                    embedding = embedding[:384]
                    embeddings.append(embedding)
                return np.array(embeddings)
        embedder = SimpleEmbedder()
    
    # Load model configuration
    model_config = load_model_config()
    if model_config is None:
        print("Error: model_config.json not found in model directory. Run training first.")
        return
    
    print(f"Loaded model configuration: {model_config}")
    
    # Load trained models with correct dimensions
    dgi_model_path = os.path.join(MODEL_DIR, 'dgi_encoder_final.pth')
    autoencoder_model_path = os.path.join(MODEL_DIR, 'autoencoder_final.pth')
    
    # Load DGI encoder with correct input dimension
    dgi_config = model_config['dgi_encoder']
    input_dim = dgi_config['input_dim']
    encoder = DGIEncoder(input_dim).to(device)
    encoder.load_state_dict(torch.load(dgi_model_path, map_location=device))
    print(f"Loaded DGI encoder from {dgi_model_path} (input_dim: {input_dim})")
    
    # Load autoencoder with correct input dimension
    ae_config = model_config['autoencoder']
    ae_input_dim = ae_config['input_dim']
    autoencoder = ImprovedAutoencoder(ae_input_dim).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
    print(f"Loaded autoencoder from {autoencoder_model_path} (input_dim: {ae_input_dim})")
    
    # Load feature keep and scaler from model directory
    keep = load_model_artifact('feature_keep.pkl')
    if keep is None:
        print("Error: feature_keep.pkl not found in model directory. Run training first.")
        return
    
    scaler = load_model_artifact('feature_scaler.pkl')
    if scaler is None:
        print("Error: feature_scaler.pkl not found in model directory. Run training first.")
        return
    
    print(f"Loaded feature artifacts: keep mask ({np.sum(keep)} features), scaler")
    
    # Load GMM
    gmm_path = os.path.join(MODEL_DIR, 'gmm_ood.pkl')
    if os.path.exists(gmm_path):
        with open(gmm_path, 'rb') as f:
            gmm = pickle.load(f)
        print("Loaded GMM for OOD scoring.")
    else:
        gmm = None
        print("Warning: GMM not found, OOD scoring will not be used.")
    
    # Load new traces
    normal_missing = [0]
    anomaly_missing = [0]
    normal_traces_new = load_json_files(NORMAL_DIR_NEW, normal_missing)
    print(f"Loaded {len(normal_traces_new)} new normal traces (missing metadata: {normal_missing[0]})")
    anomaly_traces_new = load_json_files(ANOMALY_DIR_NEW, anomaly_missing)
    print(f"Loaded {len(anomaly_traces_new)} new anomaly traces (missing metadata: {anomaly_missing[0]})")
    
    # Convert to graphs
    normal_graphs_new = traces_to_graphs(normal_traces_new, 0)
    anomaly_graphs_new = traces_to_graphs(anomaly_traces_new, 1)
    all_graphs_new = normal_graphs_new + anomaly_graphs_new
    print(f"New graphs: normal {len(normal_graphs_new)}, anomaly {len(anomaly_graphs_new)}")
    
    # Extract motifs
    all_motifs_new = extract_motifs_for_graphs(all_graphs_new)
    
    # Aggregate embeddings
    all_embeddings_new = aggregate_embeddings(encoder, all_motifs_new, embedder, device)
    
    # Extract features
    X_raw_new = extract_feature_matrix(all_graphs_new, embedder)
    X_features_new, _ = remove_low_variance_features(X_raw_new, keep=keep)
    X_features_new, _ = check_and_normalize_features(X_features_new, scaler)
    
    # Combine
    X_new = np.hstack([X_features_new, all_embeddings_new])
    
    # Compute scores
    print(f"\nComputing anomaly scores for {len(X_new)} samples...")
    all_scores = compute_anomaly_scores(autoencoder, gmm, X_new, device)
    print(f"Score range: {all_scores.min():.4f} to {all_scores.max():.4f}")
    
    # Evaluate
    y_true = np.array([0] * len(normal_graphs_new) + [1] * len(anomaly_graphs_new))
    results = evaluate_anomaly_detection(y_true, all_scores)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS ON NEW DATASET")
    print(f"{'='*60}")
    print(f"Dataset: {len(normal_graphs_new)} normal, {len(anomaly_graphs_new)} anomaly samples")
    print(f"Model: {model_config['autoencoder']['input_dim']}-dim autoencoder")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"{'='*60}")
    for k, v in results.items():
        if k != 'threshold':
            print(f"{k.upper()}: {v:.4f}")
    
    # Save results
    eval_results = {
        'dataset_info': {
            'normal_samples': len(normal_graphs_new),
            'anomaly_samples': len(anomaly_graphs_new),
            'total_samples': len(normal_graphs_new) + len(anomaly_graphs_new)
        },
        'model_info': model_config,
        'evaluation_results': results,
        'scores': all_scores.tolist(),
        'true_labels': y_true.tolist()
    }
    
    results_path = os.path.join(EVAL_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")
    
    # Save detailed per-sample results
    detailed_results = []
    for i in range(len(y_true)):
        pred = 1 if all_scores[i] >= results['threshold'] else 0
        detailed_results.append({
            'sample_id': i,
            'true_label': int(y_true[i]),
            'predicted_label': pred,
            'anomaly_score': float(all_scores[i]),
            'is_correct': bool(y_true[i] == pred)
        })
    
    detailed_path = os.path.join(EVAL_DIR, 'detailed_results.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed per-sample results saved to {detailed_path}")
    
    # Save summary statistics
    summary_stats = {
        'evaluation_summary': {
            'total_samples': len(y_true),
            'normal_samples': int(np.sum(y_true == 0)),
            'anomaly_samples': int(np.sum(y_true == 1)),
            'correct_predictions': int(sum([r['is_correct'] for r in detailed_results])),
            'incorrect_predictions': int(sum([not r['is_correct'] for r in detailed_results])),
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1']),
            'auc_roc': float(results['auc_roc']),
            'auc_pr': float(results['auc_pr']),
            'optimal_threshold': float(results['threshold'])
        },
        'score_statistics': {
            'min_score': float(all_scores.min()),
            'max_score': float(all_scores.max()),
            'mean_score': float(all_scores.mean()),
            'std_score': float(all_scores.std()),
            'median_score': float(np.median(all_scores))
        }
    }
    
    summary_path = os.path.join(EVAL_DIR, 'summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Summary statistics saved to {summary_path}")
    
    # Print per-sample results for first few samples
    print(f"\nPer-sample results (first 10):")
    print(f"{'Idx':<5} {'True':<5} {'Score':<10} {'Pred':<5}")
    print("-" * 30)
    for i in range(min(10, len(y_true))):
        pred = 1 if all_scores[i] >= results['threshold'] else 0
        print(f"{i:<5} {y_true[i]:<5} {all_scores[i]:<10.4f} {pred:<5}")

if __name__ == "__main__":
    main()