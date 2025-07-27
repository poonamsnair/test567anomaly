# --- Imports and Configuration ---
import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Optional, Any
import networkx as nx
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROD_DATA_BASE_DIR = "prod_dataset"
PROD_DATA_SUBFOLDERS = ["internlm_agent", "snorkelai_agent-finance-reasoning"]
MAX_TOTAL_FILES = 1000
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
SCORING_METHOD = "ae_loss" # Options: "ae_loss", "mahalanobis"
PCA_COMPONENTS = 45  # Dynamic based on dataset size
DGI_HIDDEN_DIM = 256
DGI_NUM_LAYERS = 2
EPS = 1e-7

# --- Output Directory Configuration ---
def get_output_directory(dataset_name: str, run_number: int = 1) -> str:
    base_dir = f"{dataset_name}_output"
    run_dir = f"run_{run_number}"
    output_dir = os.path.join(base_dir, run_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_output_paths(dataset_name: str, run_number: int = 1) -> tuple[str, str, str]:
    output_dir = get_output_directory(dataset_name, run_number)
    model_path = os.path.join(output_dir, "best_dgi_anomaly_model.pt")
    scores_path = os.path.join(output_dir, "best_dgi_anomaly_detection_results.csv")
    curve_path = os.path.join(output_dir, "best_dgi_pr_curve_data.npz")
    return model_path, scores_path, curve_path

# --- Import Universal Trace Schema ---
from universal_trace_schema import UniversalTraceSchema, TRANSFORMER_NAME

# --- Initialize Core Components ---
print("--- Initializing Core Components ---")
embed_model = SentenceTransformer(TRANSFORMER_NAME)
print(f"Loaded Sentence Transformer: {TRANSFORMER_NAME}")
pca = PCA(n_components=PCA_COMPONENTS)
scaler = StandardScaler()
print("--- Core Components Initialized ---")

# --- DGI Model Components ---
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, out_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(out_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

class AvgReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq, batch):
        return global_mean_pool(seq, batch)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, batch):
        try:
            c_x = c[batch]
            sc_1 = self.f_k(h_pl, c_x).squeeze(1)
            sc_2 = self.f_k(h_mi, c_x).squeeze(1)
            logits = torch.cat((sc_1, sc_2))
            return logits
        except Exception as e:
            logger.error(f"Discriminator forward error: {e}")
            return torch.zeros(2 * h_pl.size(0), device=h_pl.device, dtype=h_pl.dtype)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, num_layers=DGI_NUM_LAYERS):
        super().__init__()
        self.gcn = GCN(n_in, n_h, num_layers=num_layers)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, x, edge_index, batch):
        try:
            h_1 = self.gcn(x, edge_index)
        except Exception as e:
            logger.error(f"DGI GCN positive pass error: {e}")
            h_1 = torch.zeros((x.size(0), self.gcn.convs[-1].out_channels), device=x.device, dtype=x.dtype)

        try:
            c = self.read(h_1, batch)
            c = self.sigm(c)
        except Exception as e:
            logger.error(f"DGI readout/sigmoid error: {e}")
            num_graphs = torch.max(batch).item() + 1 if batch.numel() > 0 else 1
            c = torch.zeros((num_graphs, h_1.size(1) if h_1.numel() > 0 else DGI_HIDDEN_DIM), device=x.device, dtype=x.dtype)

        try:
            x_shuf = x[torch.randperm(x.size(0))]
            h_2 = self.gcn(x_shuf, edge_index)
        except Exception as e:
            logger.error(f"DGI GCN negative pass error: {e}")
            h_2 = torch.zeros_like(h_1)

        try:
            ret = self.disc(c, h_1, h_2, batch)
        except Exception as e:
            logger.error(f"DGI discriminator call error: {e}")
            ret = torch.zeros(2 * h_1.size(0), device=x.device, dtype=x.dtype)

        return ret, h_1

    def embed(self, x, edge_index, batch):
        try:
            h_1 = self.gcn(x, edge_index)
            c = self.read(h_1, batch)
            return h_1.detach(), c.detach()
        except Exception as e:
            logger.error(f"DGI embed error: {e}")
            h_dim = self.gcn.convs[-1].out_channels if self.gcn.convs else DGI_HIDDEN_DIM
            h_1 = torch.zeros((x.size(0), h_dim), device=x.device, dtype=x.dtype)
            c = torch.zeros((torch.max(batch).item() + 1 if batch.numel() > 0 else 1, h_dim), device=x.device, dtype=x.dtype)
            return h_1.detach(), c.detach()

class GraphLevelAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim // 4, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

    def forward(self, z):
        try:
            z_b = self.enc(z)
            z_r = self.dec(z_b)
            return z_r
        except Exception as e:
            logger.error(f"GraphLevelAE forward error: {e}")
            return torch.zeros_like(z)

class DGIAEModel(nn.Module):
    def __init__(self, in_dim, hid=DGI_HIDDEN_DIM):
        super().__init__()
        self.dgi = DGI(in_dim, hid)
        self.ae = GraphLevelAE(hid)

    def forward(self, x, edge_index, batch, edge_attr=None):
        try:
            logits, h_1 = self.dgi(x, edge_index, batch)
            z = global_mean_pool(h_1, batch)
            z_r = self.ae(z)
            loss_ae = F.mse_loss(z_r, z)
            labels = torch.cat((torch.ones(h_1.size(0)), torch.zeros(h_1.size(0)))).to(DEVICE)
            logits_clamped = torch.clamp(torch.sigmoid(logits), min=EPS, max=1.0-EPS)
            loss_dgi = F.binary_cross_entropy(logits_clamped, labels)
            total_loss = 1.0 * loss_dgi + 0.5 * loss_ae
            return total_loss, loss_ae, z
        except Exception as e:
            logger.error(f"DGIAEModel forward error: {e}")
            num_graphs = torch.max(batch).item() + 1 if batch.numel() > 0 else 1
            return torch.tensor(0.0, device=x.device, requires_grad=True), torch.tensor(0.0, device=x.device, requires_grad=True), torch.zeros((num_graphs, self.dgi.gcn.convs[-1].out_channels if self.dgi.gcn.convs else DGI_HIDDEN_DIM), device=x.device, dtype=x.dtype)

# --- Training and Evaluation Functions ---
def train_epoch(model, loader, opt):
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(loader, desc="Training Batches")
    for batch in pbar:
        try:
            batch = batch.to(DEVICE)
            if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index') or not hasattr(batch, 'batch'):
                logger.warning("Skipping batch with missing attributes.")
                continue
            loss, _, _ = model(batch.x, batch.edge_index, batch.batch)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or Inf loss encountered, skipping batch.")
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Avg Loss': f'{total_loss / num_batches:.4f}'})
        except Exception as e:
            logger.error(f"Error processing training batch: {e}")
            continue
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Completed Training Epoch. Average Loss: {avg_loss:.4f}")
    return avg_loss

# --- Scoring Functions ---
def score_ae_loss(model, data):
    model.eval()
    with torch.no_grad():
        try:
            data = data.to(DEVICE)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
            _, loss_ae, _ = model(data.x, data.edge_index, batch_index)
            return loss_ae.item()
        except Exception as e:
            logger.error(f"AE scoring error: {e}")
            return float('inf')

def score_mahalanobis(model, data, mahalanobis_scorer):
    model.eval()
    with torch.no_grad():
        try:
            data = data.to(DEVICE)
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
            _, _, z = model(data.x, data.edge_index, batch_index)
            z_np = z.cpu().numpy().flatten()
            return mahalanobis_scorer.score(z_np)
        except Exception as e:
            logger.error(f"Mahalanobis scoring error: {e}")
            return float('inf')

class MahalanobisScorer:
    def __init__(self):
        self.mean = None
        self.cov_inv = None
        self.fitted = False

    def fit(self, z_vectors):
        if len(z_vectors) == 0:
            logger.error("No normal representations for Mahalanobis fitting.")
            return
        try:
            z_np = np.array(z_vectors)
            self.mean = np.mean(z_np, axis=0)
            cov = np.cov(z_np, rowvar=False)
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            self.cov_inv = np.linalg.inv(cov_reg)
            self.fitted = True
            logger.info("Mahalanobis scorer fitted.")
        except np.linalg.LinAlgError:
            logger.error("Covariance matrix inversion failed.")
            self.fitted = False
        except Exception as e:
             logger.error(f"Mahalanobis fitting error: {e}")
             self.fitted = False

    def score(self, z_vector):
        if not self.fitted:
            logger.warning("Mahalanobis scorer not fitted.")
            return float('inf')
        try:
            diff = z_vector - self.mean
            dist_sq = np.dot(np.dot(diff, self.cov_inv), diff)
            return np.sqrt(dist_sq)
        except Exception as e:
             logger.error(f"Mahalanobis distance calculation error: {e}")
             return float('inf')

def is_anomaly(data):
    return int(bool(data.get("errors")))

def optimize_threshold(y_true, y_scores):
    try:
        prec, rec, th = precision_recall_curve(y_true, y_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        idx = np.argmax(f1)
        return th[idx] if idx < len(th) else (th[-1] if len(th) > 0 else 0.5)
    except Exception as e:
        logger.error(f"Threshold optimization error: {e}")
        return 0.5

def build_graph_from_schema(schema: UniversalTraceSchema):
    try:
        G = schema.build_graph(embed_model)
        if len(G.nodes()) == 0:
            logger.warning("Empty graph built.")
            return Data()

        node_features_list = []
        node_ids_sorted = sorted(G.nodes())
        for node_id in node_ids_sorted:
            if 'features' in G.nodes[node_id]:
                node_features_list.append(G.nodes[node_id]['features'])
            else:
                logger.warning(f"Node {node_id} missing 'features'.")
                return Data()
        if not node_features_list:
             logger.warning("No node features extracted.")
             return Data()
        x_base = torch.tensor(np.stack(node_features_list), dtype=torch.float)

        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        node_type_counts = {}
        for node_id in G.nodes():
            nt = G.nodes[node_id]['node_type']
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
        initial_input_len = len(schema.initial_input) if schema.initial_input else 0
        final_output_len = len(schema.final_output) if schema.final_output else 0
        num_resources = len(schema.resources_used)

        graph_stats = np.array([
            float(num_nodes), float(num_edges),
            float(node_type_counts.get('user', 0)),
            float(node_type_counts.get('agent', 0)),
            float(node_type_counts.get('tool', 0)),
            float(initial_input_len),
            float(final_output_len),
            float(num_resources)
        ], dtype=np.float32)

        graph_stats_repeated = np.tile(graph_stats, (x_base.size(0), 1))
        x_enhanced = torch.cat([x_base, torch.tensor(graph_stats_repeated, dtype=torch.float)], dim=1)

        edge_list = list(G.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attrs = [G.edges[src, tgt]['edge_type_id'] for src, tgt in edge_list]
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        data = Data(x=x_enhanced, edge_index=edge_index, edge_attr=edge_attr)
        return data
    except Exception as e:
        logger.error(f"PyG data building error: {e}")
        return Data()

# --- Main Execution Pipeline ---
def main(dataset_name: str = "internlm_agent", run_number: int = 1, datasets_to_load: List[str] = None):
    # If no specific datasets provided, use the dataset_name as the only dataset
    if datasets_to_load is None:
        datasets_to_load = [dataset_name]
    
    MODEL_SAVE_PATH, SCORES_SAVE_PATH, CURVE_SAVE_PATH = get_output_paths(dataset_name, run_number)
    print("\n" + "="*60)
    print("  DGI Anomaly Detection Pipeline")
    print("="*60)
    print(f"Configuration Summary:")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Run Number: {run_number}")
    print(f"  - Datasets to Load: {datasets_to_load}")
    print(f"  - Output Directory: {get_output_directory(dataset_name, run_number)}")
    print(f"  - Scoring Method: '{SCORING_METHOD}'")
    print(f"  - DGI GCN Layers: {DGI_NUM_LAYERS}")
    print(f"  - Max Files to Load: {MAX_TOTAL_FILES}")
    print(f"  - Device: {DEVICE}")
    print("="*60 + "\n")

    # --- 1. Load Traces and Determine Labels ---
    print("--- Step 1: Loading Data ---")
    traces_and_data = []
    count = 0
    print(f"Searching for JSON files in {PROD_DATA_BASE_DIR}...")
    for subfolder in datasets_to_load:
        folder_path = os.path.join(PROD_DATA_BASE_DIR, subfolder)
        if not os.path.exists(folder_path):
            logger.warning(f"Subfolder {folder_path} does not exist.")
            continue
        json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
        print(f"  Found {len(json_files)} files in '{subfolder}'. Loading...")
        for file_path in tqdm(json_files, desc=f"  Loading {subfolder}", leave=False):
            if count >= MAX_TOTAL_FILES:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                uts = UniversalTraceSchema.from_dict(data)
                if uts is not None:
                    traces_and_data.append((uts, data))
                    count += 1
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        if count >= MAX_TOTAL_FILES:
            break
    print(f"  Successfully loaded {len(traces_and_data)} valid traces.")
    print("--- Step 1 Complete ---\n")

    # --- 2. Fit PCA on Normal Training Data Embeddings ---
    print("--- Step 2: Fitting PCA (Dimensionality Reduction) ---")
    try:
        print("  Extracting embeddings from normal traces for PCA fitting...")
        normal_embeddings_for_pca = []
        for trace, original_data in tqdm(traces_and_data, desc="  Extracting Normal Embeddings", leave=False):
            if not is_anomaly(original_data):
                try:
                    G = trace.build_graph(embed_model)
                    if len(G.nodes()) > 0:
                        node_embeddings = embed_model.encode([n.content for n in trace.nodes])
                        normal_embeddings_for_pca.append(node_embeddings)
                except Exception as e:
                    logger.error(f"PCA embedding extraction error: {e}")
                    continue
        if normal_embeddings_for_pca:
            try:
                all_normal_emb = np.vstack(normal_embeddings_for_pca)
                if all_normal_emb.size > 0:
                    print(f"  Fitting PCA on {all_normal_emb.shape[0]} samples (dim {all_normal_emb.shape[1]} -> {PCA_COMPONENTS})...")
                    pca.fit(all_normal_emb)
                    print("  PCA fitted successfully.")
                else:
                    logger.error("No valid normal embeddings for PCA.")
                    return
            except Exception as e:
                logger.error(f"PCA fitting error: {e}")
                return
        else:
            logger.error("No normal samples for PCA.")
            return
    except Exception as e:
         logger.error(f"PCA preparation error: {e}")
         return
    print("--- Step 2 Complete ---\n")

    # --- 3. Build Graph Dataset with Labels and Process Features ---
    print("--- Step 3: Building Graphs and Processing Features ---")
    graphs = []
    labels = []
    all_processed_features_list = []
    print("  Converting traces to graphs and adding features...")
    for trace, original_data in tqdm(traces_and_data, desc="  Building Graphs", leave=False):
        try:
            pyg_data = build_graph_from_schema(trace)
            if pyg_data.x is None or pyg_data.x.numel() == 0:
                logger.warning("Skipping graph with no features.")
                continue
            features_np = pyg_data.x.numpy()
            raw_emb_dim = embed_model.get_sentence_embedding_dimension()
            num_graph_stats = 8
            if features_np.shape[1] >= raw_emb_dim + 2 + num_graph_stats:
                raw_emb_part = features_np[:, :raw_emb_dim]
                other_part = features_np[:, raw_emb_dim:]
                emb_pca = pca.transform(raw_emb_part)
                processed_features = np.concatenate([emb_pca, other_part], axis=1)
                all_processed_features_list.append(processed_features)
                pyg_data.x = torch.tensor(processed_features, dtype=torch.float)
                # Store only the anomaly label instead of the entire original data
                pyg_data.anomaly_label = is_anomaly(original_data)
                graphs.append(pyg_data)
                labels.append(is_anomaly(original_data))
            else:
                logger.warning(f"Feature dim too small: {features_np.shape[1]}, need {raw_emb_dim + 2 + num_graph_stats}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Graph processing error: {e}")
            continue
    if not graphs:
        logger.error("No graphs created. Exiting.")
        return
    print(f"  Created {len(graphs)} graphs with labels.")

    print("  Fitting StandardScaler for feature normalization...")
    if all_processed_features_list:
        try:
            all_processed_features_np = np.vstack(all_processed_features_list)
            if all_processed_features_np.size > 0:
                scaler.fit(all_processed_features_np)
                print(f"  Scaler fitted on features of shape {all_processed_features_np.shape}")
            else:
                logger.error("No processed features to fit scaler.")
                return
        except Exception as e:
            logger.error(f"Scaler fitting error: {e}")
            return
    else:
        logger.error("No processed features list.")
        return

    print("  Normalizing graph features...")
    scaled_graphs = []
    for g in tqdm(graphs, desc="  Scaling Features", leave=False):
        try:
            scaled_features_np = scaler.transform(g.x.numpy())
            g.x = torch.tensor(scaled_features_np, dtype=torch.float)
            g.y = torch.tensor([g.anomaly_label])
            scaled_graphs.append(g)
        except Exception as e:
            logger.error(f"Feature scaling error: {e}")
            continue
    graphs = scaled_graphs
    if not graphs:
         logger.error("No graphs after scaling. Exiting.")
         return
    labels = [g.y.item() for g in graphs]
    print(f"  Final dataset size after scaling: {len(graphs)} graphs.")
    print("--- Step 3 Complete ---\n")

    # --- 4. Split Dataset ---
    print("--- Step 4: Splitting Dataset ---")
    try:
        num_anomalies = sum(labels)
        num_normals = len(labels) - num_anomalies
        print(f"  Dataset Composition: {len(labels)} Total, {num_normals} Normal, {num_anomalies} Anomaly")
        if num_anomalies == 0:
            logger.error("No anomalies found.")
            return
        desired_test_anomalies = max(1, num_anomalies // 4)
        desired_test_size = min(0.5, max(0.1, desired_test_anomalies / len(labels)))
        print(f"  Attempting test split (~{desired_test_size*100:.1f}%) for ~{desired_test_anomalies} anomalies in test.")
        temp_graphs, test_graphs, temp_labels, test_labels = train_test_split(
            graphs, labels, test_size=desired_test_size, random_state=42, stratify=labels
        )
        temp_num_anomalies = sum(temp_labels)
        temp_num_normals = len(temp_labels) - temp_num_anomalies
        print(f"  Temp set for train/val: {len(temp_labels)} Total, {temp_num_normals} Normal, {temp_num_anomalies} Anomaly")
        if temp_num_anomalies == 0:
             logger.error("No anomalies in temp set.")
             return
        desired_val_anomalies = max(1, temp_num_anomalies // 4)
        desired_val_size = min(0.5, max(0.1, desired_val_anomalies / len(temp_labels)))
        print(f"  Attempting val split (~{desired_val_size*100:.1f}%) for ~{desired_val_anomalies} anomalies in val.")
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            temp_graphs, temp_labels, test_size=desired_val_size, random_state=42, stratify=temp_labels
        )
        train_normal_graphs = [g for g, l in zip(train_graphs, train_labels) if l == 0]
        print(f"  Final Splits:")
        print(f"    - Train (Normal Only, for DGI): {len(train_normal_graphs)} graphs")
        print(f"    - Validation: {len(val_graphs)} graphs (Anomalies: {sum(val_labels)})")
        print(f"    - Test: {len(test_graphs)} graphs (Anomalies: {sum(test_labels)})")
        total_anomalies_in_splits = sum(val_labels) + sum(test_labels) + (len(train_graphs) - len(train_normal_graphs))
        if len(train_normal_graphs) > 0 and total_anomalies_in_splits > 0:
            ratio = len(train_normal_graphs) / total_anomalies_in_splits
            print(f"  Normal:Anomaly ratio in splits: {ratio:.2f}:1")
        else:
            print("  Could not calculate Normal:Anomaly ratio.")
    except ValueError as e:
        logger.error(f"Stratified split error: {e}")
        logger.info("Attempting simple split...")
        try:
             temp_graphs, test_graphs, temp_labels, test_labels = train_test_split(
                 graphs, labels, test_size=0.2, random_state=42
             )
             train_graphs, val_graphs, train_labels, val_labels = train_test_split(
                 temp_graphs, temp_labels, test_size=0.25, random_state=42
             )
             train_normal_graphs = [g for g, l in zip(train_graphs, train_labels) if l == 0]
             print(f"  Simple Split Results:")
             print(f"    - Train Normal: {len(train_normal_graphs)}")
             print(f"    - Validation: {len(val_graphs)} (Anomalies: {sum(val_labels)})")
             print(f"    - Test: {len(test_graphs)} (Anomalies: {sum(test_labels)})")
        except Exception as e:
             logger.error(f"Simple split failed: {e}")
             return
    except Exception as e:
         logger.error(f"Train/test split error: {e}")
         return
    print("--- Step 4 Complete ---\n")

    # --- 5. Create Data Loaders ---
    print("--- Step 5: Creating Data Loaders ---")
    if not train_normal_graphs:
        logger.error("No normal samples for training.")
        return
    print(f"  Creating DataLoader for training (Normal data only)...")
    train_loader = DataLoader(train_normal_graphs, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  Creating DataLoader for validation...")
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Creating DataLoader for testing (batch size 1)...")
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    print("--- Step 5 Complete ---\n")

    # --- 6. Initialize Model ---
    print("--- Step 6: Initializing Model ---")
    input_dim = PCA_COMPONENTS + 2 + 8
    print(f"  Creating DGI+AE model with input dimension {input_dim}...")
    model = DGIAEModel(in_dim=input_dim, hid=DGI_HIDDEN_DIM).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model created. Total parameters: {total_params}, Trainable: {trainable_params}")
    print(f"  Initializing AdamW optimizer...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print("--- Step 6 Complete ---\n")

    # --- 7. Train Model ---
    print("--- Step 7: Training Model ---")
    best_loss = float('inf')
    patience_counter = 0
    print(f"  Starting training loop for up to {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        print(f"  Epoch {epoch}/{EPOCHS}:")
        train_loss = train_epoch(model, train_loader, opt)
        model.eval()
        val_loss = 0
        num_val_batches = 0
        print(f"    Validating...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"    Validating Epoch {epoch}", leave=False):
                try:
                    batch = batch.to(DEVICE)
                    if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index') or not hasattr(batch, 'batch'):
                        continue
                    loss, _, _ = model(batch.x, batch.edge_index, batch.batch)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        num_val_batches += 1
                except Exception as e:
                    logger.error(f"Validation batch error: {e}")
                    continue
        val_loss /= max(num_val_batches, 1)
        print(f"    Validation Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"    Model checkpoint saved (best val loss: {best_loss:.4f}).")
            except Exception as e:
                logger.error(f"Model save error: {e}")
        else:
            patience_counter += 1
            print(f"    No improvement for {patience_counter} epochs.")
            if patience_counter >= PATIENCE:
                print(f"    Early stopping triggered after epoch {epoch}.")
                break
    print("--- Step 7 Complete ---\n")

    # --- 8. Load Best Model and Evaluate ---
    print("--- Step 8: Evaluating Best Model ---")
    print("  Loading best model checkpoint...")
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print("  Best model loaded successfully.")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return

    current_scoring_method = SCORING_METHOD
    if SCORING_METHOD == "mahalanobis":
        print("  Preparing Mahalanobis scorer using normal validation data...")
        normal_z_vectors = []
        model.eval()
        print("    Extracting graph representations `z` from normal validation data...")
        with torch.no_grad():
            for data in tqdm(val_graphs, desc="    Extracting Normal Z", leave=False):
                if not data.anomaly_label:
                    try:
                        data = data.to(DEVICE)
                        batch_index = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
                        _, _, z = model(data.x, data.edge_index, batch_index)
                        z_np = z.cpu().numpy().flatten()
                        normal_z_vectors.append(z_np)
                    except Exception as e:
                        logger.error(f"Z extraction error: {e}")
                        continue
        mahalanobis_scorer = MahalanobisScorer()
        print("    Fitting Mahalanobis distribution...")
        mahalanobis_scorer.fit(normal_z_vectors)
        if not mahalanobis_scorer.fitted:
            logger.error("Mahalanobis scorer failed. Falling back to AE loss.")
            current_scoring_method = "ae_loss"
        else:
            print("    Mahalanobis scorer fitted.")
            current_scoring_method = SCORING_METHOD

    print(f"  Scoring validation data for threshold optimization using '{current_scoring_method}'...")
    val_scores = []
    for data in tqdm(val_graphs, desc=f"  Val Scoring ({current_scoring_method})", leave=False):
        try:
            if current_scoring_method == "ae_loss":
                val_scores.append(score_ae_loss(model, data))
            elif current_scoring_method == "mahalanobis":
                val_scores.append(score_mahalanobis(model, data, mahalanobis_scorer))
        except Exception as e:
            logger.error(f"Val scoring error: {e}")
            val_scores.append(float('inf'))
    th = optimize_threshold(val_labels, val_scores)
    print(f"  Optimized threshold: {th:.4f}")

    print(f"  Scoring test data using '{current_scoring_method}'...")
    test_scores = []
    for data in tqdm(test_graphs, desc=f"  Test Scoring ({current_scoring_method})", leave=False):
        try:
            if current_scoring_method == "ae_loss":
                test_scores.append(score_ae_loss(model, data))
            elif current_scoring_method == "mahalanobis":
                test_scores.append(score_mahalanobis(model, data, mahalanobis_scorer))
        except Exception as e:
            logger.error(f"Test scoring error: {e}")
            test_scores.append(float('inf'))

    processed_test_scores = np.nan_to_num(test_scores, nan=1e6, posinf=1e6, neginf=0.0)
    y_true = test_labels
    y_pred = (np.array(processed_test_scores) > th).astype(int)

    print("  Calculating final performance metrics...")
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        auc_roc = roc_auc_score(y_true, processed_test_scores)
        pr, re, _ = precision_recall_curve(y_true, processed_test_scores)
        auc_pr = auc(re, pr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
        acc, prec, rec, f1, f2, auc_roc, auc_pr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        pr, re = np.array([]), np.array([])
        tn, fp, fn, tp = 0, 0, 0, 0

    print("\n" + "="*60)
    print("  FINAL EVALUATION RESULTS (Test Set)")
    print("="*60)
    print(f"Scoring Method Used: {current_scoring_method}")
    print(f"Detection Threshold: {th:.4f}")
    print("-"*40)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"F2 Score          : {f2:.4f}")
    print(f"AUC-ROC           : {auc_roc:.4f}")
    print(f"AUC-PR (Preferred): {auc_pr:.4f}")
    print("-"*40)
    print("Confusion Matrix (TN FP / FN TP):")
    print(f"[ {tn:4d} {fp:4d} ]")
    print(f"[ {fn:4d} {tp:4d} ]")
    print("-"*40)
    print(f"Model saved to       : {MODEL_SAVE_PATH}")

    try:
        df = pd.DataFrame({
            "graph_index": range(len(test_graphs)),
            "true_label": y_true,
            "anomaly_score": processed_test_scores,
            "predicted_label": y_pred
        })
        df.to_csv(SCORES_SAVE_PATH, index=False)
        print(f"Per-graph scores saved to: {SCORES_SAVE_PATH}")
    except Exception as e:
        logger.error(f"CSV save error: {e}")

    try:
        if len(pr) > 0 and len(re) > 0:
            np.savez(CURVE_SAVE_PATH, precision=pr, recall=re)
            print(f"PR curve data saved to: {CURVE_SAVE_PATH}")
    except Exception as e:
        logger.error(f"PR curve save error: {e}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    dataset_name = "internlm_agent"
    run_number = 1
    datasets_to_load = None
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        # If dataset_name is provided, use it as the only dataset to load
        datasets_to_load = [dataset_name]
    
    if len(sys.argv) > 2:
        try:
            run_number = int(sys.argv[2])
        except ValueError:
            print(f"Invalid run number: {sys.argv[2]}. Using default run number 1.")
            run_number = 1
    
    # Allow specifying multiple datasets as additional arguments
    if len(sys.argv) > 3:
        additional_datasets = sys.argv[3:]
        if datasets_to_load is None:
            datasets_to_load = additional_datasets
        else:
            datasets_to_load.extend(additional_datasets)
    
    main(dataset_name, run_number, datasets_to_load)
