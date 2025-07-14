import os
import json
import glob
import random
import warnings
import argparse
import hashlib
from datetime import datetime
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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
from torch.utils.data import DataLoader as TorchDataLoader
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import shap
import umap
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
import re
from matplotlib.patches import Ellipse

warnings.filterwarnings('ignore', category=RuntimeWarning)

DIG_ROOT = 'training'
MODEL_DIR = os.path.join(DIG_ROOT, 'model')
CHECKPOINT_DIR = os.path.join(DIG_ROOT, 'checkpoints')
VISUALS_DIR = os.path.join(DIG_ROOT, 'visuals')
CACHE_DIR = os.path.join(DIG_ROOT, 'cache')
NORMAL_DIR = 'training_dataset/normal/'
ANOMALY_DIR = 'training_dataset/anomaly/'
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

CACHE_VERSION = "v1.2"
CACHE_METADATA_FILE = "cache_metadata.json"

class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR, prefix="", clear_cache=False):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.metadata_file = os.path.join(cache_dir, CACHE_METADATA_FILE)
        os.makedirs(cache_dir, exist_ok=True)
        if clear_cache:
            self.clear_all_caches()
        self.metadata = self._load_metadata()
        if 'version' not in self.metadata or self.metadata['version'] != CACHE_VERSION:
            self.clear_all_caches()
            self.metadata['version'] = CACHE_VERSION
            self.metadata['created'] = datetime.now().isoformat()
            self._save_metadata()
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        return {}
    
    def _save_metadata(self):
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def _get_cache_path(self, filename):
        prefixed_filename = f"{self.prefix}_{filename}" if self.prefix else filename
        return os.path.join(self.cache_dir, prefixed_filename)
    
    def _get_cache_hash(self, filename):
        cache_path = self._get_cache_path(filename)
        if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
        return None
    
    def load_cache(self, filename, validate=True):
        cache_path = self._get_cache_path(filename)
        if not os.path.exists(cache_path):
            return None
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if validate:
                current_hash = self._get_cache_hash(filename)
                stored_hash = self.metadata.get('hashes', {}).get(filename)
                if stored_hash and current_hash != stored_hash:
                    return None
            return data
    
    def save_cache(self, data, filename, validate=True):
        cache_path = self._get_cache_path(filename)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        if validate:
            if 'hashes' not in self.metadata:
                self.metadata['hashes'] = {}
            self.metadata['hashes'][filename] = self._get_cache_hash(filename)
            self._save_metadata()
    
    def clear_all_caches(self):
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl') and (not self.prefix or filename.startswith(self.prefix)):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
        if 'hashes' in self.metadata:
            keys_to_remove = [k for k in self.metadata['hashes'] if (not self.prefix or k.startswith(self.prefix))]
            for key in keys_to_remove:
                del self.metadata['hashes'][key]
            self._save_metadata()
    
    def get_cache_info(self):
        info = {
            'cache_dir': self.cache_dir,
            'prefix': self.prefix,
            'version': self.metadata.get('version', 'unknown'),
            'created': self.metadata.get('created', 'unknown'),
            'cached_files': []
        }
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl') and (not self.prefix or filename.startswith(self.prefix)):
                    filepath = os.path.join(self.cache_dir, filename)
                    size = os.path.getsize(filepath)
                    info['cached_files'].append({
                        'filename': filename,
                        'size_bytes': size,
                        'size_mb': size / (1024 * 1024)
                    })
        return info
    
    def validate_all_caches(self):
        valid = True
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl') and (not self.prefix or filename.startswith(self.prefix)):
                base_filename = filename[len(self.prefix)+1:] if self.prefix else filename
                current_hash = self._get_cache_hash(filename)
                stored_hash = self.metadata.get('hashes', {}).get(base_filename)
                if stored_hash and current_hash != stored_hash:
                    valid = False
        return valid

cache_manager = None

def get_cache_manager(prefix="", clear_cache=False):
    global cache_manager
    if cache_manager is None or cache_manager.prefix != prefix:
        cache_manager = CacheManager(CACHE_DIR, prefix, clear_cache)
    return cache_manager

motif_embed_cache = {}

def motif_to_pyg_cached(subg, embedder, device='cpu'):
    h = motif_hash(subg)
    if h in motif_embed_cache:
        data = motif_embed_cache[h]
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        return data
    node_texts = [subg.nodes[n]['text'] for n in subg.nodes]
    print(f"[Semantic Embedding] Encoding node texts for motif with {len(node_texts)} nodes.")
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
        print(f"[Semantic Embedding] Encoding step texts for trace with question: '{question[:50]}...'")
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

def motif_hash(subg):
    edges = sorted([tuple(sorted(e)) for e in subg.edges])
    node_texts = tuple(sorted([subg.nodes[n]['text'] for n in subg.nodes]))
    return hash((tuple(edges), node_texts))

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
    print(f"[Semantic Embedding] Aggregating motif embeddings for {len(motifs_list)} graphs...")
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

class DGIEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] + [GCNConv(hidden_dim, output_dim)])
        self.proj = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
        self.discriminator = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    
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
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.3, denoising=True):
        super().__init__()
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

def train_dgi_encoder(encoder, train_loader, device, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    def edge_dropout(edge_index, drop_prob=0.2):
        if edge_index.size(1) == 0 or drop_prob <= 0:
            return edge_index
        keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
        return edge_index[:, keep_mask]

    print(f"🔄 Training DGI encoder for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pos_z = encoder.project(encoder(batch.x, batch.edge_index, batch.batch))
            neg_edge_index = edge_dropout(batch.edge_index)
            neg_z = encoder.project(encoder(batch.x, neg_edge_index, batch.batch))
            pos_score = encoder.discriminate(pos_z)
            neg_score = encoder.discriminate(neg_z)
            loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.6f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(encoder, optimizer, epoch, avg_loss, 'dgi_encoder.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"   Early stopping at epoch {epoch}")
            break
    print(f"✅ DGI training completed. Best loss: {best_loss:.6f}")
    return encoder, losses

def train_autoencoder(autoencoder, train_data, device, epochs=500, lr=1e-4, batch_size=64):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    train_tensor = torch.FloatTensor(train_data).to(device)
    train_loader = TorchDataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    print(f"🔄 Training autoencoder for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for x in train_loader:
            optimizer.zero_grad()
            decoded, _ = autoencoder(x)
            loss = F.mse_loss(decoded, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.6f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(autoencoder, optimizer, epoch, avg_loss, 'autoencoder.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"   Early stopping at epoch {epoch}")
            break
    print(f"✅ Autoencoder training completed. Best loss: {best_loss:.6f}")

    # Compute training latents for OOD
    autoencoder.eval()
    with torch.no_grad():
        train_latents = []
        train_loader = TorchDataLoader(torch.FloatTensor(train_data), batch_size=batch_size, shuffle=False)
        for x in train_loader:
            _, encoded = autoencoder(x)
            train_latents.append(encoded.cpu().numpy())
        train_latents = np.concatenate(train_latents, axis=0)
    
    # Fit GMM for density estimation (OOD)
    gmm = GaussianMixture(n_components=5, random_state=SEED)  # Adjust n_components based on data
    gmm.fit(train_latents)
    
    # Save GMM
    with open(os.path.join(MODEL_DIR, 'gmm_ood.pkl'), 'wb') as f:
        pickle.dump(gmm, f)
    
    return autoencoder, losses

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, filename))

def verify_model_files():
    required_files = [
        'dgi_encoder_final.pth',
        'autoencoder_final.pth', 
        'feature_scaler.pkl',
        'feature_keep.pkl',
        'model_config.json'
    ]
    missing_files = [filename for filename in required_files if not os.path.exists(os.path.join(MODEL_DIR, filename))]
    if missing_files:
        return False
        return True

def compare_with_previous_runs(current_run_file, visuals_dir):
    with open(current_run_file, 'r') as f:
        current_results = json.load(f)
    run_files = [f for f in glob.glob(os.path.join(visuals_dir, 'run_results_*.json')) if f != current_run_file]
    if not run_files:
        return
    comparisons = []
    for run_file in run_files:
        with open(run_file, 'r') as f:
            prev_results = json.load(f)
        comparison = {
            'previous_run': os.path.basename(run_file),
            'previous_prefix': prev_results.get('cache_prefix', 'unknown'),
            'previous_timestamp': prev_results.get('timestamp', 'unknown'),
            'metric_changes': {}
        }
        current_metrics = current_results.get('results', {})
        prev_metrics = prev_results.get('results', {})
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
            if metric in current_metrics and metric in prev_metrics:
                change = current_metrics[metric] - prev_metrics[metric]
                comparison['metric_changes'][metric] = {
                    'current': current_metrics[metric],
                    'previous': prev_metrics[metric],
                    'change': change,
                    'change_percent': (change / prev_metrics[metric] * 100) if prev_metrics[metric] != 0 else 0
                }
        comparisons.append(comparison)
    if comparisons:
        comparison_file = os.path.join(visuals_dir, f'run_comparison_{current_results["cache_prefix"]}.json')
        with open(comparison_file, 'w') as f:
            json.dump({'current_run': current_results, 'comparisons': comparisons}, f, indent=2)

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

def compute_threshold_from_train_scores(train_scores, method='mean_plus_3std'):
    if method == 'mean_plus_3std':
        threshold = np.mean(train_scores) + 3 * np.std(train_scores)
    elif method == 'percentile_95':
        threshold = np.percentile(train_scores, 95)
    elif method == 'percentile_99':
        threshold = np.percentile(train_scores, 99)
    elif method == 'mean_plus_2std':
        threshold = np.mean(train_scores) + 2 * np.std(train_scores)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return threshold

def evaluate_anomaly_detection_with_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_pr': average_precision_score(y_true, y_scores),
        'auc_roc': roc_auc_score(y_true, y_scores),
        'threshold': threshold,
        'threshold_method': 'pre_computed'
    }

def generate_visualizations(dgi_losses, ae_losses, X_train, X_test, results, all_scores, y_true, normal_train_embeddings, normal_test_embeddings, anomaly_embeddings, normal_train_motifs, anomaly_motifs, normal_test_graphs, anomaly_graphs, embedder, autoencoder, encoder, device, keep, scaler, normal_train_graphs=None, gmm=None):
    print(f"   📊 Creating training loss curves...")
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    # Training loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(dgi_losses, color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('DGI Training Loss', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(VISUALS_DIR, 'dgi_loss_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ae_losses, color='green', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Autoencoder Training Loss', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(VISUALS_DIR, 'autoencoder_loss_curve.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating feature correlation heatmap...")
    # Feature correlation heatmap
    feature_names = [
        'num_steps', 'duration_minutes', 'num_tools', 'num_agents',
        'avg_step_length', 'max_step_length', 'min_step_length', 'step_length_std', 'unique_agents',
        'agent_entropy', 'tool_entropy', 'tool_usage_frequency',
        'steps_per_minute', 'question_length', 'step_complexity', 'repetitive_actions',
        'tool_diversity', 'step_consistency', 'agent_switching_frequency', 'tool_usage_pattern',
        'question_complexity', 'has_long_steps', 'has_short_steps', 'role_entropy', 'type_entropy',
        'consecutive_repeat_count', 'tool_agent_overlap', 'avg_step_duration',
        'question_tool_match', 'content_embedding_std', 'max_agent_streak',
        'contains_error', 'contains_timeout', 'contains_failed', 'contains_exception'
    ]
    scalar_feature_count = len(feature_names)
    X_train_scalar = X_train[:, :scalar_feature_count]
    corr_matrix = np.corrcoef(X_train_scalar.T)
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', square=True, 
                xticklabels=feature_names, yticklabels=feature_names, cbar_kws={'shrink': 0.8}, linewidths=0.5)
    plt.title('Feature Correlation Heatmap (Scalar Features)', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'feature_correlation_heatmap.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating confusion matrix...")
    # Confusion matrix
    y_pred = (all_scores >= results['threshold']).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title('Confusion Matrix', fontsize=14)
    plt.grid(False)
    plt.savefig(os.path.join(VISUALS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating performance plots...")
    # Results plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fpr, tpr, _ = roc_curve(y_true, all_scores)
    axs[0, 0].plot(fpr, tpr, color='darkorange', linewidth=2, label=f"AUC = {results['auc_roc']:.3f}")
    axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axs[0, 0].set_title('ROC Curve', fontsize=12)
    axs[0, 0].legend(fontsize=10)
    precision, recall, _ = precision_recall_curve(y_true, all_scores)
    axs[0, 1].plot(recall, precision, color='purple', linewidth=2, label=f"AUC = {results['auc_pr']:.3f}")
    axs[0, 1].set_title('PR Curve', fontsize=12)
    axs[0, 1].legend(fontsize=10)
    
    # Improved score distribution with separate subplots
    normal_scores = all_scores[y_true == 0]
    anomaly_scores = all_scores[y_true == 1]
    
    # Normal scores subplot
    axs[1, 0].hist(normal_scores, bins=30, color='blue', alpha=0.7, label=f'Normal (n={len(normal_scores)})')
    axs[1, 0].axvline(results['threshold'], color='red', ls='--', label=f'Threshold: {results["threshold"]:.3f}')
    axs[1, 0].axvline(np.mean(normal_scores), color='green', ls='-', alpha=0.7, label=f'Mean: {np.mean(normal_scores):.3f}')
    axs[1, 0].set_title('Score Distribution - Normal vs Anomaly', fontsize=12)
    axs[1, 0].set_xlabel('Anomaly Score', fontsize=10)
    axs[1, 0].set_ylabel('Frequency', fontsize=10)
    axs[1, 0].legend(fontsize=9)
    
    # Overlay anomaly scores on the same plot
    axs[1, 0].hist(anomaly_scores, bins=30, color='red', alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})')
    axs[1, 0].legend(fontsize=9)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [results[m] for m in metrics]
    bars = axs[1, 1].bar(metrics, values, color='skyblue')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title('Performance Metrics', fontsize=12)
    for bar, v in zip(bars, values):
        axs[1, 1].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'anomaly_detection_results.png'), dpi=300)
    plt.close()
    
    # Create a separate detailed score distribution plot
    print(f"   📊 Creating detailed score distribution...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Normal scores
    ax1.hist(normal_scores, bins=50, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(results['threshold'], color='red', ls='--', linewidth=2, label=f'Threshold: {results["threshold"]:.3f}')
    ax1.axvline(np.mean(normal_scores), color='green', ls='-', linewidth=2, label=f'Mean: {np.mean(normal_scores):.3f}')
    ax1.axvline(np.percentile(normal_scores, 95), color='orange', ls=':', linewidth=2, label=f'95th percentile: {np.percentile(normal_scores, 95):.3f}')
    ax1.set_title(f'Normal Scores Distribution (n={len(normal_scores)})', fontsize=14)
    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Anomaly scores
    ax2.hist(anomaly_scores, bins=50, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(results['threshold'], color='red', ls='--', linewidth=2, label=f'Threshold: {results["threshold"]:.3f}')
    ax2.axvline(np.mean(anomaly_scores), color='green', ls='-', linewidth=2, label=f'Mean: {np.mean(anomaly_scores):.3f}')
    ax2.axvline(np.percentile(anomaly_scores, 5), color='orange', ls=':', linewidth=2, label=f'5th percentile: {np.percentile(anomaly_scores, 5):.3f}')
    ax2.set_title(f'Anomaly Scores Distribution (n={len(anomaly_scores)})', fontsize=14)
    ax2.set_xlabel('Anomaly Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'detailed_score_distribution.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating embedding projections...")
    # Embedding projections
    try:
        reducer = umap.UMAP(random_state=SEED)
        method = 'UMAP'
    except ImportError:
        reducer = TSNE(random_state=SEED)
        method = 't-SNE'
    combined_embeddings = np.vstack([normal_train_embeddings, normal_test_embeddings, anomaly_embeddings])
    combined_labels = np.concatenate([np.zeros(len(normal_train_embeddings)), np.ones(len(normal_test_embeddings)), np.full(len(anomaly_embeddings), 2)])
    emb_2d = reducer.fit_transform(combined_embeddings)
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    labels = ['Train Normal', 'Test Normal', 'Test Anomaly']
    for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
        mask = combined_labels == i
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, marker=marker, s=60, alpha=0.8, label=label, edgecolors='gray')
    plt.xlabel(f'{method} Component 1', fontsize=12)
    plt.ylabel(f'{method} Component 2', fontsize=12)
    plt.title(f'{method} Projection of Graph Embeddings', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'embeddings_projection.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating cluster analysis...")
    # Cluster analysis for anomaly embeddings
    if len(anomaly_embeddings) > 2:
        range_n_clusters = range(2, min(7, len(anomaly_embeddings)))
        silhouette_avgs = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
            cluster_labels = kmeans.fit_predict(anomaly_embeddings)
            silhouette_avg = silhouette_score(anomaly_embeddings, cluster_labels)
            silhouette_avgs.append(silhouette_avg)
        plt.figure(figsize=(10, 6))
        plt.plot(list(range_n_clusters), silhouette_avgs, marker='o', color='purple', linewidth=2)
        plt.xlabel('Number of clusters', fontsize=12)
        plt.ylabel('Average Silhouette Score', fontsize=12)
        plt.title('Silhouette Analysis for KMeans (Anomaly Embeddings)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'anomaly_silhouette.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        Z = linkage(anomaly_embeddings, method='ward')
        dendrogram(Z, truncate_mode='level', p=5)
        plt.title('Hierarchical Clustering Dendrogram (Anomaly Embeddings)', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'anomaly_dendrogram.png'), dpi=300)
        plt.close()

    print(f"   📊 Creating SHAP explanations...")
    # SHAP explanations
    X_train_sample = X_train[np.random.choice(len(X_train), min(100, len(X_train)), replace=False)]
    X_test_sample = X_test[np.random.choice(len(X_test), min(100, len(X_test)), replace=False)]
    
    def get_feature_names(num_features):
        scalar_features = [
            'num_steps', 'duration_minutes', 'num_tools', 'num_agents',
            'avg_step_length', 'max_step_length', 'min_step_length', 'step_length_std', 'unique_agents',
            'agent_entropy', 'tool_entropy', 'tool_usage_frequency',
            'steps_per_minute', 'question_length', 'step_complexity', 'repetitive_actions',
            'tool_diversity', 'step_consistency', 'agent_switching_frequency', 'tool_usage_pattern',
            'question_complexity', 'has_long_steps', 'has_short_steps', 'role_entropy', 'type_entropy',
            'consecutive_repeat_count', 'tool_agent_overlap', 'avg_step_duration',
            'question_tool_match', 'content_embedding_std', 'max_agent_streak',
            'contains_error', 'contains_timeout', 'contains_failed', 'contains_exception'
        ]
        vector_features = [f'content_embed_{i}' for i in range(384)]
        motif_features = [f'motif_embed_{i}' for i in range(64)]
        all_features = scalar_features + vector_features + motif_features
        return all_features[:num_features]
    
    def ae_predict(x):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            decoded, _ = autoencoder(x)
            return ((decoded - x)**2).mean(dim=1).cpu().numpy()
    
    explainer = shap.KernelExplainer(ae_predict, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample, nsamples=100)
    feature_names = get_feature_names(X_test_sample.shape[1])
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'shap_explanations.png'), dpi=300)
    plt.close()

    feature_explanations = {
        'num_steps': 'Number of steps in the trace',
        'duration_minutes': 'Total duration in minutes',
        'num_tools': 'Number of unique tools used',
        'num_agents': 'Number of unique agents involved',
        'avg_step_length': 'Average length of step content',
        'max_step_length': 'Maximum length of any step',
        'min_step_length': 'Minimum length of any step',
        'step_length_std': 'Standard deviation of step lengths',
        'unique_agents': 'Number of unique agents',
        'agent_entropy': 'Entropy of agent distribution',
        'tool_entropy': 'Entropy of tool usage distribution',
        'tool_usage_frequency': 'Frequency of tool usage per step',
        'steps_per_minute': 'Steps executed per minute',
        'question_length': 'Length of the original question',
        'step_complexity': 'Average word count per step',
        'repetitive_actions': 'Number of repetitive agent actions',
        'tool_diversity': 'Diversity of tools used',
        'step_consistency': 'Consistency of step lengths',
        'agent_switching_frequency': 'How often agents switch',
        'tool_usage_pattern': 'Hash of tool usage pattern',
        'question_complexity': 'Word count of question',
        'has_long_steps': 'Whether trace has very long steps',
        'has_short_steps': 'Whether trace has very short steps',
        'role_entropy': 'Entropy of role distribution',
        'type_entropy': 'Entropy of step type distribution',
        'consecutive_repeat_count': 'Number of consecutive repeated steps',
        'tool_agent_overlap': 'Overlap between tools and agents',
        'avg_step_duration': 'Average time per step',
        'question_tool_match': 'Whether question mentions tools used',
        'content_embedding_std': 'Standard deviation of content embeddings',
        'max_agent_streak': 'Longest streak of same agent',
        'contains_error': 'Whether trace contains error keywords',
        'contains_timeout': 'Whether trace contains timeout keywords',
        'contains_failed': 'Whether trace contains failed keywords',
        'contains_exception': 'Whether trace contains exception keywords'
    }
    with open(os.path.join(VISUALS_DIR, 'feature_explanations.txt'), 'w') as f:
        f.write("Feature Explanations for SHAP Analysis\n")
        f.write("=" * 50 + "\n\n")
        for i, feature_name in enumerate(feature_names):
            if feature_name in feature_explanations:
                f.write(f"{i:2d}. {feature_name}: {feature_explanations[feature_name]}\n")
            elif feature_name.startswith('content_embed_'):
                f.write(f"{i:2d}. {feature_name}: Content embedding dimension {feature_name.split('_')[-1]}\n")
            elif feature_name.startswith('motif_embed_'):
                f.write(f"{i:2d}. {feature_name}: Motif embedding dimension {feature_name.split('_')[-1]}\n")
            else:
                f.write(f"{i:2d}. {feature_name}: Unknown feature\n")

    print(f"   📊 Creating cluster explainability...")
    # Cluster explainability
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    cluster_labels = kmeans.fit_predict(np.vstack([normal_test_embeddings, anomaly_embeddings]))
    cluster_sizes = np.bincount(cluster_labels)
    cluster_feature_means = []
    cluster_type_counts = []
    all_graphs = normal_test_graphs + anomaly_graphs
    for i in range(n_clusters):
        idxs = np.where(cluster_labels == i)[0]
        if len(idxs) == 0:
            cluster_feature_means.append(np.zeros(X_test.shape[1]))
            cluster_type_counts.append({})
            continue
        cluster_feature_means.append(X_test[idxs].mean(axis=0))
        types = []
        for idx in idxs:
            types_val = all_graphs[idx]['graph'].graph.get('anomaly_types', [])
            if isinstance(types_val, list):
                types.extend(types_val)
            elif isinstance(types_val, str) and types_val != 'unknown':
                types.append(types_val)
    cluster_type_counts.append(dict(Counter(types)))
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), cluster_sizes, color='skyblue')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Trajectories', fontsize=12)
    plt.title('Cluster Sizes (KMeans on Embeddings)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'cluster_sizes.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating cluster type distributions...")
    for i, type_count in enumerate(cluster_type_counts):
        if not type_count:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = list(type_count.keys())
        sizes = list(type_count.values())
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_weight('bold')
        ax.set_title(f'Anomaly Type Distribution in Cluster {i}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(wedges, labels, title="Anomaly Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9, title_fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, f'cluster_{i}_types.png'), dpi=300)
        plt.close()

    print(f"   📊 Creating latent space visualization...")
    # Latent space visualization
    autoencoder.eval()
    with torch.no_grad():
        latent_loader = TorchDataLoader(torch.FloatTensor(X_test), batch_size=256, shuffle=False)
        latent_vecs = []
        for x in latent_loader:
            x = x.to(device)
            _, encoded = autoencoder(x)
            latent_vecs.append(encoded.cpu().numpy())
        latent_vecs = np.concatenate(latent_vecs, axis=0)
    latent_2d = reducer.fit_transform(latent_vecs)
    plt.figure(figsize=(12, 8))
    colors = ['green', 'red']
    markers = ['s', '^']
    labels = ['Test Normal', 'Test Anomaly']
    for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
        mask = (y_true == i)
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], c=color, marker=marker, s=60, alpha=0.8, label=label, edgecolors='gray')
    plt.xlabel(f'{method} Component 1', fontsize=12)
    plt.ylabel(f'{method} Component 2', fontsize=12)
    plt.title(f'{method} Projection of Autoencoder Latent Space', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'latent_projection.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating score timeline...")
    # Score timeline
    def compute_cumulative_recon_error(autoencoder, graph, embedder, encoder, device):
        step_texts = [graph['graph'].nodes[n]['text'] for n in graph['graph'].nodes]
        if not step_texts:
            return []
        step_embeds = embedder.encode(step_texts, show_progress_bar=False)
        step_embeds = torch.tensor(step_embeds, dtype=torch.float, device=device)
        edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        batch = torch.zeros(step_embeds.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            motif_embeds = encoder(step_embeds, edge_index, batch)
        cumulative_errors = []
        for i in range(1, len(step_texts)+1):
            fake_graph = {'metadata': graph['graph'].graph, 'steps': [graph['graph'].nodes[n] for n in list(graph['graph'].nodes)[:i]], 'question': graph['graph'].graph.get('question', '')}
            features = extract_trace_features(fake_graph, embedder)
            scalar_features = [v for k, v in features.items() if k != 'content_embedding_mean' and isinstance(v, (int, float, bool))]
            vector_features = features.get('content_embedding_mean', [0.0] * 384)
            numeric_features = np.array(scalar_features + vector_features, dtype=np.float32)
            numeric_features = numeric_features[keep]
            numeric_features = scaler.transform([numeric_features])[0]
            # Fix: Add bounds checking for motif_embeds
            if i-1 < motif_embeds.shape[0]:
                motif_vec = motif_embeds[i-1].cpu().numpy()
            else:
                # Use the last available motif embedding if index is out of bounds
                motif_vec = motif_embeds[-1].cpu().numpy()
            full_vec = np.concatenate([numeric_features, motif_vec])
            x = torch.FloatTensor(full_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                decoded, _ = autoencoder(x)
                recon_error = F.mse_loss(decoded, x, reduction='mean').item()
            cumulative_errors.append(recon_error)
        return cumulative_errors

    num_examples = 3
    plt.figure(figsize=(12, 8))
    for idx, g in enumerate(normal_test_graphs[:num_examples]):
        errors = compute_cumulative_recon_error(autoencoder, g, embedder, encoder, device)
        plt.plot(range(1, len(errors)+1), errors, label=f'Normal {idx+1}', color='green', alpha=0.7, linewidth=2)
    for idx, g in enumerate(anomaly_graphs[:num_examples]):
        errors = compute_cumulative_recon_error(autoencoder, g, embedder, encoder, device)
        plt.plot(range(1, len(errors)+1), errors, label=f'Anomaly {idx+1}', color='red', alpha=0.7, linewidth=2)
    plt.xlabel('Step Number', fontsize=12)
    plt.ylabel('Cumulative Reconstruction Error', fontsize=12)
    plt.title('Anomaly Score Timeline for Example Trajectories', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'score_timeline_examples.png'), dpi=300)
    plt.close()

    print(f"   📊 Creating 3D trajectory plots...")
    # 3D Trajectory Plot
    if normal_train_graphs is not None:
        all_graphs = normal_train_graphs + normal_test_graphs + anomaly_graphs
        all_labels = [0] * len(normal_train_graphs) + [1] * len(normal_test_graphs) + [2] * len(anomaly_graphs)
    else:
        all_graphs = normal_test_graphs + anomaly_graphs
        all_labels = [1] * len(normal_test_graphs) + [2] * len(anomaly_graphs)
    def plot_3d_trajectories(graphs, labels, title, filename, embedder, max_trajectories=20):
        if len(graphs) > max_trajectories:
            indices = np.random.choice(len(graphs), max_trajectories, replace=False)
            graphs = [graphs[i] for i in indices]
            labels = [labels[i] for i in indices]
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'green', 'red']
        for i, (graph_item, label) in enumerate(zip(graphs, labels)):
            G = graph_item['graph']
            nodes = list(G.nodes())
            if len(nodes) < 3:
                continue  # Skip trajectories with fewer than 3 steps
            node_texts = [G.nodes[n]['text'] for n in nodes]
            node_embeds = embedder.encode(node_texts, show_progress_bar=False)
            pca = PCA(n_components=3)
            node_embeds_3d = pca.fit_transform(node_embeds)
            x, y, z = node_embeds_3d[:, 0], node_embeds_3d[:, 1], node_embeds_3d[:, 2]
            ax.plot(x, y, z, color=colors[label], alpha=0.7, linewidth=2)
            ax.scatter(x[0], y[0], z[0], c=colors[label], marker='o', s=100, alpha=0.8, edgecolors='black')
            ax.scatter(x[-1], y[-1], z[-1], c=colors[label], marker='s', s=100, alpha=0.8, edgecolors='black')
            if i < 5:
                for j, (x_pos, y_pos, z_pos) in enumerate(zip(x, y, z)):
                    ax.text(x_pos, y_pos, z_pos, f'{j+1}', fontsize=8, alpha=0.8)
        ax.set_xlabel('PCA Component 1', fontsize=12)
        ax.set_ylabel('PCA Component 2', fontsize=12)
        ax.set_zlabel('PCA Component 3', fontsize=12)
        ax.set_title(title, fontsize=14)
        handles = [plt.Line2D([0], [0], color=colors[i], label=label) for i, label in enumerate(['Train Normal', 'Test Normal', 'Test Anomaly'])]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, filename), dpi=300)
        plt.close()
    plot_3d_trajectories(all_graphs, all_labels, '3D Trajectory Visualization: All Sequences', '3d_trajectories_all.png', embedder, max_trajectories=30)
    plot_3d_trajectories(anomaly_graphs, [2] * len(anomaly_graphs), '3D Trajectory Visualization: Anomaly Sequences Only', '3d_trajectories_anomalies.png', embedder, max_trajectories=15)
    normal_graphs = normal_test_graphs[:10]
    anomaly_graphs_sample = anomaly_graphs[:10]
    comparison_graphs = normal_graphs + anomaly_graphs_sample
    comparison_labels = [1] * len(normal_graphs) + [2] * len(anomaly_graphs_sample)
    plot_3d_trajectories(comparison_graphs, comparison_labels, '3D Trajectory Comparison: Normal vs Anomaly', '3d_trajectories_comparison.png', embedder, max_trajectories=20)

    print(f"   📊 Creating anomaly types analysis...")
    # Anomaly types distribution
    anomaly_types_count = {}
    anomaly_types_detected = {}
    anomaly_scores = all_scores[len(normal_test_graphs):]
    anomaly_predictions = (anomaly_scores >= results['threshold']).astype(int)
    for i, g in enumerate(anomaly_graphs):
        anomaly_types = g['graph'].graph.get('anomaly_types', [])
        detected = anomaly_predictions[i]
        if isinstance(anomaly_types, list):
            for anomaly_type in anomaly_types:
                anomaly_types_count[anomaly_type] = anomaly_types_count.get(anomaly_type, 0) + 1
                anomaly_types_detected[anomaly_type] = anomaly_types_detected.get(anomaly_type, 0) + detected
        elif isinstance(anomaly_types, str) and anomaly_types != 'unknown':
            anomaly_types_count[anomaly_types] = anomaly_types_count.get(anomaly_types, 0) + 1
            anomaly_types_detected[anomaly_types] = anomaly_types_detected.get(anomaly_types, 0) + detected
    if anomaly_types_count:
        sorted_pairs = sorted(anomaly_types_count.items(), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_pairs)
        rates = [(anomaly_types_detected.get(t, 0) / c * 100) for t, c in sorted_pairs]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        bars1 = ax1.bar(range(len(types)), counts, color='coral', alpha=0.7)
        ax1.set_xlabel('Anomaly Types', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of Anomaly Types', fontsize=14)
        ax1.set_xticks(range(len(types)))
        ax1.set_xticklabels(types, rotation=45, ha='right', fontsize=10)
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
        bars2 = ax2.bar(range(len(types)), rates, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Anomaly Types', fontsize=12)
        ax2.set_ylabel('Detection Rate (%)', fontsize=12)
        ax2.set_title('Detection Accuracy by Anomaly Type', fontsize=14)
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels(types, rotation=45, ha='right', fontsize=10)
        ax2.set_ylim(0, 100)
        for bar, rate in zip(bars2, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Detection Rate')
        ax2.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'anomaly_types_analysis.png'), dpi=300)
        plt.close()

    # GMM Contour/Ellipse Plot over Latent Space
    print(f"   📊 Creating GMM contour/ellipse plot...")
    def plot_gmm_ellipses(gmm, ax, colors=None):
        for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            if covar.shape == (2, 2):
                v, w = np.linalg.eigh(covar)
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = np.degrees(angle)
                ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=colors[i % len(colors)] if colors else None, alpha=0.25, lw=2, edgecolor=colors[i % len(colors)] if colors else 'black', facecolor='none')
                ax.add_patch(ell)
    # Project GMM means and covariances to 2D
    # Get the same transformation as used for latent_2d
    # Use the reducer (UMAP or t-SNE) fitted above
    # Project all test latent vectors
    # Fit GMM on full latent_vecs if not already
    # Use gmm from pipeline (should be fitted on train latents, but for visualization, use test latents)
    # Project GMM means and covariances to 2D
    if hasattr(gmm, 'means_') and gmm.means_.shape[1] == latent_vecs.shape[1]:
        gmm_means_2d = reducer.transform(gmm.means_)
        gmm_covars_2d = []
        for cov in gmm.covariances_:
            if cov.shape[0] > 2:
                # Project covariance to 2D using the same reducer components (if available)
                if hasattr(reducer, 'embedding_') and hasattr(reducer, 'components_'):
                    W = reducer.components_[:2, :]
                    cov2d = W @ cov @ W.T
                elif hasattr(reducer, 'components_'):
                    W = reducer.components_[:2, :]
                    cov2d = W @ cov @ W.T
                else:
                    cov2d = np.eye(2)
            else:
                cov2d = cov
            gmm_covars_2d.append(cov2d)
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (color, marker, label) in enumerate(zip(colors, markers, labels)):
            mask = (y_true == i)
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], c=color, marker=marker, s=60, alpha=0.8, label=label, edgecolors='gray')
        for i, (mean, covar) in enumerate(zip(gmm_means_2d, gmm_covars_2d)):
            if covar.shape == (2, 2):
                v, w = np.linalg.eigh(covar)
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = np.degrees(angle)
                ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=colors[i % len(colors)], alpha=0.25, lw=2, edgecolor=colors[i % len(colors)], facecolor='none')
                ax.add_patch(ell)
        ax.set_xlabel(f'{method} Component 1', fontsize=12)
        ax.set_ylabel(f'{method} Component 2', fontsize=12)
        ax.set_title(f'GMM Components in Latent Space ({method})', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'gmm_latent_contours.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--cache-prefix', type=str, default='')
    parser.add_argument('--validate-cache', action='store_true')
    parser.add_argument('--threshold-method', type=str, default='mean_plus_3std', 
                        choices=['mean_plus_2std', 'mean_plus_3std', 'percentile_95', 'percentile_99'])
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"🚀 Starting Anomaly Detection Pipeline...")
    print(f"📱 Using device: {device}")
    print(f"⚙️  Cache setting: {'Disabled (--no-cache)' if args.no_cache else 'Enabled'}")
    
    cache_mgr = get_cache_manager(prefix=args.cache_prefix, clear_cache=args.clear_cache)
    
    print(f"\n📂 Loading trace data...")
    normal_missing = [0]
    anomaly_missing = [0]
    normal_traces = load_json_files(NORMAL_DIR, normal_missing)
    print(f"✅ Loaded {len(normal_traces)} normal traces (missing metadata: {normal_missing[0]})")
    anomaly_traces = load_json_files(ANOMALY_DIR, anomaly_missing)
    print(f"✅ Loaded {len(anomaly_traces)} anomaly traces (missing metadata: {anomaly_missing[0]})")
    
    print(f"\n🔄 Splitting normal data into train/test...")
    normal_train_traces, normal_test_traces = train_test_split(normal_traces, test_size=0.2, random_state=SEED)
    print(f"✅ Normal train: {len(normal_train_traces)}, test: {len(normal_test_traces)}")
    
    def get_or_compute(cache_file, compute_func):
        if args.no_cache:
            print(f"🔄 Computing {cache_file} (no-cache mode)...")
            return compute_func()
        else:
            cached_data = cache_mgr.load_cache(cache_file)
            if cached_data is not None:
                print(f"📂 Loaded cached {cache_file}")
                return cached_data
            else:
                print(f"🔄 Computing {cache_file}...")
            data = compute_func()
            cache_mgr.save_cache(data, cache_file)
        return data
    
    print(f"\n🔄 Converting traces to graphs...")
    normal_train_graphs = get_or_compute('normal_train_graphs.pkl', lambda: traces_to_graphs(normal_train_traces, 0))
    normal_test_graphs = get_or_compute('normal_test_graphs.pkl', lambda: traces_to_graphs(normal_test_traces, 0))
    anomaly_graphs = get_or_compute('anomaly_graphs.pkl', lambda: traces_to_graphs(anomaly_traces, 1))
    print(f"✅ Graphs: train {len(normal_train_graphs)}, test {len(normal_test_graphs)}, anomaly {len(anomaly_graphs)}")
    
    print(f"\n🔄 Loading sentence embedder...")
    embedder = SentenceTransformer('models/all-MiniLM-L6-v2')
    print(f"✅ SentenceTransformer loaded successfully")
    
    print(f"\n🔄 Extracting motifs from graphs...")
    normal_train_motifs = get_or_compute('normal_train_motifs.pkl', lambda: extract_motifs_for_graphs(normal_train_graphs))
    normal_test_motifs = get_or_compute('normal_test_motifs.pkl', lambda: extract_motifs_for_graphs(normal_test_graphs))
    anomaly_motifs = get_or_compute('anomaly_motifs.pkl', lambda: extract_motifs_for_graphs(anomaly_graphs))
    print(f"✅ Motifs extracted: train {len(normal_train_motifs)}, test {len(normal_test_motifs)}, anomaly {len(anomaly_motifs)}")
    
    print(f"\n🔄 Preparing DGI training data...")
    train_data_list = [motif_to_pyg_cached(motif, embedder, device) for motifs in normal_train_motifs for motif in motifs]
    input_dim = train_data_list[0].x.size(1) if train_data_list else 384
    print(f"✅ Prepared {len(train_data_list)} motif samples for DGI training")
    
    print(f"\n🔄 Initializing DGI encoder...")
    dgi_model_path = os.path.join(MODEL_DIR, 'dgi_encoder_final.pth')
    encoder = DGIEncoder(input_dim).to(device)
    if not args.no_cache and os.path.exists(dgi_model_path):
        encoder.load_state_dict(torch.load(dgi_model_path, map_location=device))
        print(f"✅ Loaded DGI encoder from {dgi_model_path}")
        dgi_losses = []
    else:
        print(f"🔄 Training DGI encoder...")
        train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE_MOTIF, shuffle=True)
        encoder, dgi_losses = train_dgi_encoder(encoder, train_loader, device)
        torch.save(encoder.state_dict(), dgi_model_path)
        print(f"✅ DGI encoder training completed and saved")
    
    print(f"\n🔄 Computing graph embeddings...")
    normal_train_embeddings = get_or_compute('normal_train_embeddings.pkl', lambda: aggregate_embeddings(encoder, normal_train_motifs, embedder, device))
    normal_test_embeddings = get_or_compute('normal_test_embeddings.pkl', lambda: aggregate_embeddings(encoder, normal_test_motifs, embedder, device))
    anomaly_embeddings = get_or_compute('anomaly_embeddings.pkl', lambda: aggregate_embeddings(encoder, anomaly_motifs, embedder, device))
    print(f"✅ Embeddings computed: train {normal_train_embeddings.shape}, test {normal_test_embeddings.shape}, anomaly {anomaly_embeddings.shape}")
    
    print(f"\n🔄 Extracting feature matrices...")
    X_train_raw = get_or_compute('X_train_raw.pkl', lambda: extract_feature_matrix(normal_train_graphs, embedder))
    X_train_features, keep = remove_low_variance_features(X_train_raw)
    if not args.no_cache:
        cache_mgr.save_cache(keep, 'feature_keep.pkl')
    X_test_raw = get_or_compute('X_test_raw.pkl', lambda: extract_feature_matrix(normal_test_graphs + anomaly_graphs, embedder))
    X_test_features, _ = remove_low_variance_features(X_test_raw, keep=keep)
    print(f"✅ Feature matrices extracted: train {X_train_raw.shape}, test {X_test_raw.shape}")
    
    print(f"\n🔄 Normalizing features...")
    X_train_features, scaler = check_and_normalize_features(X_train_features)
    X_test_features, _ = check_and_normalize_features(X_test_features, scaler)
    print(f"✅ Features normalized")
    
    print(f"\n🔄 Combining features with embeddings...")
    X_train = np.hstack([X_train_features, normal_train_embeddings])
    X_test = np.hstack([X_test_features, np.vstack([normal_test_embeddings, anomaly_embeddings])])
    print(f"✅ Combined features: train {X_train.shape}, test {X_test.shape}")
    
    print(f"\n🔄 Initializing autoencoder...")
    autoencoder_model_path = os.path.join(MODEL_DIR, 'autoencoder_final.pth')
    autoencoder = ImprovedAutoencoder(X_train.shape[1]).to(device)
    if not args.no_cache and os.path.exists(autoencoder_model_path):
        autoencoder.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
        print(f"✅ Loaded autoencoder from {autoencoder_model_path}")
        ae_losses = []
    else:
        print(f"🔄 Training autoencoder...")
        autoencoder, ae_losses = train_autoencoder(autoencoder, X_train, device)
        torch.save(autoencoder.state_dict(), autoencoder_model_path)
        print(f"✅ Autoencoder training completed and saved")
    
    # Load or train GMM
    gmm_path = os.path.join(MODEL_DIR, 'gmm_ood.pkl')
    if not args.no_cache and os.path.exists(gmm_path):
        with open(gmm_path, 'rb') as f:
            gmm = pickle.load(f)
    else:
        # If training, the train_autoencoder should return gmm, but since it's not, assume it's trained here or adjust
        # For completeness, recompute if needed
        autoencoder.eval()
        with torch.no_grad():
            train_latents = []
            train_loader = TorchDataLoader(torch.FloatTensor(X_train), batch_size=64, shuffle=False)
            for x in train_loader:
                x = x.to(device)
                _, encoded = autoencoder(x)
                train_latents.append(encoded.cpu().numpy())
            train_latents = np.concatenate(train_latents, axis=0)
        gmm = GaussianMixture(n_components=5, random_state=SEED)
        gmm.fit(train_latents)
        with open(gmm_path, 'wb') as f:
            pickle.dump(gmm, f)
    
    print(f"\n🔄 Saving model artifacts...")
    with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, 'feature_keep.pkl'), 'wb') as f:
        pickle.dump(keep, f)
    model_config = {
        'dgi_encoder': {'input_dim': int(input_dim), 'hidden_dim': 128, 'output_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        'autoencoder': {'input_dim': int(X_train.shape[1]), 'hidden_dim': 256, 'latent_dim': 128, 'dropout': 0.3},
        'feature_dim': int(X_train.shape[1]),
        'embedding_dim': int(normal_train_embeddings.shape[1]) if len(normal_train_embeddings) > 0 else 64
    }
    with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✅ Model artifacts saved")
    
    print(f"\n🔄 Computing anomaly scores...")
    train_scores = compute_anomaly_scores(autoencoder, gmm, X_train, device)
    threshold = compute_threshold_from_train_scores(train_scores, method=args.threshold_method)
    test_scores = compute_anomaly_scores(autoencoder, gmm, X_test, device)
    print(f"✅ Anomaly scores computed: train {len(train_scores)}, test {len(test_scores)}")
    
    print(f"\n🔄 Evaluating model performance...")
    y_true = np.array([0] * len(normal_test_graphs) + [1] * len(anomaly_graphs))
    results = evaluate_anomaly_detection_with_threshold(y_true, test_scores, threshold)
    print(f"✅ Evaluation completed")
    
    print(f"\n🔄 Generating visualizations...")
    generate_visualizations(dgi_losses, ae_losses, X_train, X_test, results, test_scores, y_true, normal_train_embeddings, normal_test_embeddings, anomaly_embeddings, normal_train_motifs, anomaly_motifs, normal_test_graphs, anomaly_graphs, embedder, autoencoder, encoder, device, keep, scaler, normal_train_graphs, gmm)
    print(f"✅ Visualizations generated")
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📊 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1 Score: {results['f1']:.4f}")
    print(f"   AUC-ROC: {results['auc_roc']:.4f}")
    print(f"   AUC-PR: {results['auc_pr']:.4f}")
    print(f"   Threshold: {results['threshold']:.6f}")
    print(f"\n📁 Outputs saved to:")
    print(f"   Models: {MODEL_DIR}")
    print(f"   Visualizations: {VISUALS_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()