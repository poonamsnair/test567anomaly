# Technical Implementation Documentation

## Training Pipeline Architecture

```mermaid
graph TB
    %% Data Loading & Preprocessing
    subgraph "Data Loading & Preprocessing"
        A1[Load JSON Traces] --> A2[Validate Metadata]
        A2 --> A3[Extract Features]
        A3 --> A4[Split Train/Test]
        A4 --> A5[Convert to Graphs]
        A5 --> A6[NetworkX Directed Graphs]
    end
    
    subgraph "Graph Processing"
        A6 --> B1[Extract Motifs]
        B1 --> B2[Subgraph Motifs 2-4 nodes]
        B2 --> B3[PyTorch Geometric Data]
        B3 --> B4[Batch Processing]
    end
    
    subgraph "Feature Engineering"
        A3 --> C1[Statistical Features]
        A3 --> C2[Semantic Features]
        C2 --> C3[Sentence Transformers]
        C3 --> C4[Text Embeddings]
        C1 --> C5[Feature Selection]
        C4 --> C5
        C5 --> C6[Normalization]
    end
    
    subgraph "DGI Training"
        B4 --> D1[DGI Encoder]
        D1 --> D2[Contrastive Learning]
        D2 --> D3[Graph Embeddings]
        D3 --> D4[Save DGI Model]
    end
    
    subgraph "Feature Combination"
        D3 --> E1[Combine Features]
        C6 --> E1
        E1 --> E2[Combined Feature Matrix]
    end
    
    subgraph "Autoencoder Training"
        E2 --> F1[Autoencoder Model]
        F1 --> F2[Reconstruction Training]
        F2 --> F3[Latent Representations]
        F3 --> F4[Save Autoencoder]
        F3 --> F5[GMM Training]
        F5 --> F6[Density Model]
        F6 --> F7[Save GMM]
    end
    
    subgraph "Model Artifacts"
        D4 --> G1[Model Directory]
        F4 --> G1
        F7 --> G1
        G1 --> G2[dgi_encoder_final.pth]
        G1 --> G3[autoencoder_final.pth]
        G1 --> G4[gmm_ood.pkl]
        G1 --> G5[feature_scaler.pkl]
        G1 --> G6[feature_keep.pkl]
        G1 --> G7[model_config.json]
    end
    
    subgraph "Visualization & Analysis"
        F2 --> H1[Training Curves]
        F3 --> H2[Embedding Analysis]
        F6 --> H3[Density Plots]
        H1 --> H4[Save Visualizations]
        H2 --> H4
        H3 --> H4
    end
    
    %% Styling
    classDef data fill:#e3f2fd
    classDef graph fill:#f3e5f5
    classDef feature fill:#e8f5e8
    classDef dgi fill:#fff3e0
    classDef ae fill:#fce4ec
    classDef artifacts fill:#f1f8e9
    classDef viz fill:#e0f2f1
    
    class A1,A2,A3,A4,A5,A6 data
    class B1,B2,B3,B4 graph
    class C1,C2,C3,C4,C5,C6 feature
    class D1,D2,D3,D4 dgi
    class E1,E2 feature
    class F1,F2,F3,F4,F5,F6,F7 ae
    class G1,G2,G3,G4,G5,G6,G7 artifacts
    class H1,H2,H3,H4 viz
```

## Data Generation Pipeline (`generate_training_data.py`)

### Core Implementation Details

#### 1. Trace Generation Strategy

The pipeline uses a **few-shot learning approach** with GPT-4o to generate realistic agent interaction traces. The generation process follows these principles:

- **Diversity**: Varies trace length, complexity, and interaction patterns
- **Realism**: Ensures traces follow realistic agent behavior patterns
- **Consistency**: Maintains structural consistency across all generated traces
- **Anomaly Injection**: Systematically introduces specific anomaly types

#### 2. Prompt Engineering

The system uses carefully crafted prompts to ensure high-quality trace generation:

```python
NORMAL_PROMPT_TEMPLATE = '''
You are an expert at generating synthetic agent traces for a reasoning pipeline. Generate 1 synthetic trace in the following JSON schema, where:
- The user asks a normal question, possibly with follow-ups or interruptions in multi-turn conversations.
- The agent successfully completes the task, with fluid, diverse interactions.
- Vary the length: some short single-turn, some long multi-turn with user questions or interruptions.
- Show valid flows, correct handoffs to agents, accurate tool usage, planning, observations, and reasoning.
- Task completed true, no errors, diverse topics like search queries, code execution, image generation, or combinations.
'''
```

#### 3. Anomaly Injection Mechanism

For anomaly generation, the system uses **targeted prompt engineering**:

```python
ANOMALY_PROMPT_TEMPLATE_BASE = '''
Make sure the trace demonstrates the following anomaly: {category}: {description}
Include the anomaly type in metadata["anomaly_types"], possibly with related types.
'''
```

This ensures that each anomaly trace specifically demonstrates the intended anomaly type while maintaining realistic structure.

#### 4. Data Validation and Processing

The pipeline includes robust error handling and validation:

```python
def extract_json(content):
    # Strip whitespace and find JSON boundaries
    start = content.find('{')
    end = content.rfind('}') + 1
    
    if start != -1 and end != 0 and start < end:
        extracted = content[start:end]
        # Validate JSON structure
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            raise ValueError(f"Extracted content is not valid JSON")
```

#### 5. File Management

The system implements intelligent file management to avoid overwriting existing data:

```python
def get_next_id(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.json')]
    if not files:
        return 1
    ids = []
    for f in files:
        match = re.search(r'_(\d+)\.json$', f)
        if match:
            ids.append(int(match.group(1)))
    return max(ids) + 1 if ids else 1
```

## Training Pipeline (`training_pipeline.py`)

### Core Implementation Details

#### 1. Cache Management System

The pipeline implements a sophisticated caching system to optimize performance:

```python
class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR, prefix="", clear_cache=False):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.metadata_file = os.path.join(cache_dir, CACHE_METADATA_FILE)
        
    def _get_cache_hash(self, filename):
        cache_path = self._get_cache_path(filename)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        return None
```

**Key Features:**
- **Version Control**: Ensures cache consistency across pipeline versions
- **Hash Validation**: MD5-based integrity checking
- **Prefix Support**: Allows multiple experiment caches
- **Automatic Cleanup**: Clears outdated caches

#### 2. Graph Processing Pipeline

The system converts agent traces into graph representations for analysis:

```python
def traces_to_graphs(traces, label):
    graphs = []
    for trace in traces:
        features = extract_trace_features(trace)
        num_nodes = len(trace.get('steps', []))
        if num_nodes == 0:
            continue
        
        G = nx.DiGraph()
        # Add graph-level metadata
        G.graph['num_steps'] = features.get('num_steps', 0)
        G.graph['tools_called'] = len(features.get('tools_called', []))
        G.graph['agents_called'] = len(features.get('agents_called', []))
        G.graph['duration_minutes'] = features.get('duration_minutes', 0.0)
        G.graph['question'] = trace.get('question', '')
        G.graph['anomaly_types'] = trace.get('metadata', {}).get('anomaly_types', 'normal' if label == 0 else 'unknown')
        
        # Add nodes (steps)
        for i, step in enumerate(trace.get('steps', [])):
            G.add_node(i, text=step.get('content', ''), agent=step.get('agent', ''), 
                      role=step.get('role', ''), type=step.get('type', ''))
        
        # Add edges (sequential flow)
        for i in range(num_nodes-1):
            G.add_edge(i, i+1)
        
        graphs.append({'graph': G, 'label': label})
    return graphs
```

#### 3. Motif Extraction Algorithm

The system extracts subgraph motifs to capture local interaction patterns:

```python
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
```

**Motif Hashing Strategy:**
```python
def motif_hash(subg):
    edges = sorted([tuple(sorted(e)) for e in subg.edges])
    node_texts = tuple(sorted([subg.nodes[n]['text'] for n in subg.nodes]))
    return hash((tuple(edges), node_texts))
```

#### 4. Feature Engineering System

The pipeline extracts comprehensive features from traces:

```python
def extract_trace_features(trace, embedder=None):
    meta = trace.get('metadata', {})
    question = trace.get('question', '')
    steps = trace.get('steps', [])
    
    # Extract basic features
    step_texts = [s.get('content', '') for s in steps]
    agents = [s.get('agent', '') for s in steps]
    roles = [s.get('role', '') for s in steps]
    types = [s.get('type', '') for s in steps]
    
    # Compute statistical features
    features = {
        'num_steps': meta.get('num_steps', len(steps)),
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
    }
    
    return features
```

#### 5. DGI Encoder Architecture

The Deep Graph Infomax encoder implements contrastive learning for graph representation:

```python
class DGIEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # Graph convolutional layers
        self.convs = nn.ModuleList([
            GCNConv(input_dim, hidden_dim)
        ] + [
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ] + [
            GCNConv(hidden_dim, output_dim)
        ])
        
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Discriminator for positive/negative sample classification
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        # Graph convolution layers
        for conv in self.convs[:-1]:
            x = F.relu(F.dropout(conv(x, edge_index), p=self.dropout, training=self.training))
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
    
    def project(self, x):
        return self.proj(x)
    
    def discriminate(self, x):
        return self.discriminator(x)
```

#### 6. Autoencoder Architecture

The improved autoencoder includes batch normalization and denoising capabilities:

```python
class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.3, denoising=False):
        super().__init__()
        self.denoising = denoising
        
        # Encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        # Add noise during training for denoising
        if self.denoising and self.training:
            x = x + torch.randn_like(x) * 0.05
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
```

#### 7. Training Process

The training pipeline follows a multi-stage approach:

1. **DGI Training**: Contrastive learning on graph motifs
2. **Embedding Generation**: Computing graph representations
3. **Feature Combination**: Merging statistical and semantic features
4. **Autoencoder Training**: Unsupervised learning on normal data
5. **Threshold Computation**: Determining anomaly detection threshold

#### 8. Hybrid Anomaly Detection Algorithm

The system uses a **hybrid approach** combining reconstruction error and out-of-distribution (OOD) detection for robust anomaly scoring:

```python
def compute_anomaly_scores(autoencoder, gmm, data, device, ood_weight=0.4):
    """
    Compute anomaly scores using hybrid reconstruction + OOD logic.
    
    Args:
        autoencoder: Trained autoencoder model
        gmm: Gaussian Mixture Model for density estimation
        data: Input data to score
        device: Computation device
        ood_weight: Weight for OOD component (default: 0.4)
    
    Returns:
        Combined anomaly scores
    """
    autoencoder.eval()
    scores = []
    
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        for i in range(0, len(data_tensor), BATCH_SIZE_MOTIF):
            batch = data_tensor[i:i+BATCH_SIZE_MOTIF]
            decoded, encoded = autoencoder(batch)
            
            # Reconstruction-based scoring (multiple error metrics)
            reconstruction_error = F.mse_loss(decoded, batch, reduction='none').mean(dim=1)
            cosine_error = 1 - F.cosine_similarity(decoded, batch, dim=1)
            l1_error = F.l1_loss(decoded, batch, reduction='none').mean(dim=1)
            
            # Combined reconstruction score (weighted average)
            recon_score = 0.5 * reconstruction_error + 0.3 * cosine_error + 0.2 * l1_error
            
            # OOD: Negative log-likelihood from GMM (higher = more OOD)
            batch_latents = encoded.cpu().numpy()
            ood_scores = -gmm.score_samples(batch_latents)
            
            # Hybrid: Weighted combination of reconstruction and OOD scores
            combined_score = (1 - ood_weight) * recon_score.cpu().numpy() + ood_weight * ood_scores
            scores.extend(combined_score)
    
    return np.array(scores)
```

**Key Components:**

1. **Reconstruction Error Scoring:**
   - **MSE Loss**: Mean squared error between input and reconstructed data
   - **Cosine Error**: 1 - cosine similarity for directional differences
   - **L1 Error**: Mean absolute error for robustness to outliers
   - **Weighted Combination**: 50% MSE + 30% Cosine + 20% L1

2. **Out-of-Distribution (OOD) Detection:**
   - **Gaussian Mixture Model (GMM)**: Fitted on training latent representations
   - **Negative Log-Likelihood**: Higher scores indicate samples outside learned distribution
   - **Density Estimation**: Captures complex multi-modal distributions

3. **Hybrid Combination:**
   - **Configurable Weight**: `ood_weight` parameter (default: 0.4)
   - **Balanced Approach**: Combines reconstruction and distribution-based detection
   - **Robust Performance**: Handles both local reconstruction failures and global distribution shifts

**Advantages of Hybrid Approach:**
- **Comprehensive Detection**: Catches both reconstruction failures and distribution shifts
- **Robust to Noise**: Multiple error metrics reduce sensitivity to outliers
- **Configurable**: Adjustable weights allow tuning for specific use cases
- **Interpretable**: Separate reconstruction and OOD components provide insights

#### 9. Threshold Computation Methods

The system supports multiple threshold computation strategies:

```python
def compute_threshold_from_train_scores(train_scores, method='mean_plus_3std'):
    if method == 'mean_plus_2std':
        return np.mean(train_scores) + 2 * np.std(train_scores)
    elif method == 'mean_plus_3std':
        return np.mean(train_scores) + 3 * np.std(train_scores)
    elif method == 'percentile_95':
        return np.percentile(train_scores, 95)
    elif method == 'percentile_99':
        return np.percentile(train_scores, 99)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
```

#### 10. Evaluation Metrics

The system computes comprehensive evaluation metrics:

```python
def evaluate_anomaly_detection_with_threshold(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'threshold': threshold
    }
```

### Performance Optimizations

#### 1. Caching Strategy

The system implements intelligent caching to avoid recomputing expensive operations:

- **Motif Extraction**: Cached to avoid recomputing subgraph operations
- **Embedding Computation**: Cached to avoid recomputing sentence embeddings
- **Feature Matrix**: Cached to avoid recomputing feature extraction
- **Model Training**: Checkpoints saved for resuming training

#### 2. Memory Management

- **Batch Processing**: Large datasets processed in batches
- **Gradient Accumulation**: Handles large models with limited memory
- **Device Management**: Automatic CPU/GPU detection and usage

#### 3. Early Stopping

The system implements early stopping to prevent overfitting:

```python
if avg_loss < best_loss:
    best_loss = avg_loss
    save_checkpoint(encoder, optimizer, epoch, avg_loss, 'dgi_encoder.pth')
    patience_counter = 0
else:
    patience_counter += 1
if patience_counter >= EARLY_STOPPING_PATIENCE:
    print(f"   Early stopping at epoch {epoch}")
    break
```

### Visualization System

The pipeline generates comprehensive visualizations for model analysis:

1. **Training Curves**: Loss progression during training
2. **Anomaly Score Distributions**: Histograms of anomaly scores
3. **Confusion Matrix**: Classification performance visualization
4. **ROC and PR Curves**: Model discrimination ability
5. **Feature Importance**: SHAP-based feature analysis
6. **Embedding Visualizations**: t-SNE and UMAP plots
7. **Sankey Diagrams**: Flow transition analysis

### Error Handling and Robustness

The system includes comprehensive error handling:

1. **Data Validation**: Ensures trace data integrity
2. **Model Checkpointing**: Saves progress during training
3. **Graceful Degradation**: Handles missing dependencies
4. **Cache Recovery**: Recovers from cache corruption
5. **Memory Management**: Handles out-of-memory situations

This technical implementation provides a robust, scalable, and efficient anomaly detection system for multi-agent reasoning traces. 