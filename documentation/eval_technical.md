# Technical Implementation Documentation

## System Architecture Overview

```mermaid
graph TB
    %% Data Flow
    subgraph "Input Data"
        A1[Agent Traces JSON] --> A2[Trace Validation]
        A2 --> A3[Metadata Extraction]
    end
    
    subgraph "Graph Processing"
        A3 --> B1[NetworkX Graph Conversion]
        B1 --> B2[Node Features]
        B1 --> B3[Edge Features]
        B2 --> B4[Motif Extraction]
        B3 --> B4
        B4 --> B5[Subgraph Motifs]
    end
    
    subgraph "Feature Engineering"
        A3 --> C1[Statistical Features]
        A3 --> C2[Semantic Features]
        C2 --> C3[Sentence Transformers]
        C3 --> C4[Text Embeddings]
        C1 --> C5[Feature Matrix]
        C4 --> C5
    end
    
    subgraph "Model Training"
        B5 --> D1[DGI Encoder]
        D1 --> D2[Graph Embeddings]
        D2 --> D3[Feature Combination]
        C5 --> D3
        D3 --> D4[Combined Features]
        D4 --> D5[Autoencoder]
        D5 --> D6[Reconstruction Model]
        D5 --> D7[Latent Representations]
        D7 --> D8[GMM Training]
        D8 --> D9[Density Model]
    end
    
    subgraph "Hybrid Anomaly Detection"
        D6 --> E1[Reconstruction Error]
        D9 --> E2[OOD Detection]
        E1 --> E3[MSE Loss]
        E1 --> E4[Cosine Error]
        E1 --> E5[L1 Error]
        E3 --> E6[Weighted Reconstruction Score]
        E4 --> E6
        E5 --> E6
        E2 --> E7[Negative Log-Likelihood]
        E7 --> E8[OOD Score]
        E6 --> E9[Combined Anomaly Score]
        E8 --> E9
    end
    
    subgraph "Evaluation & Output"
        E9 --> F1[Threshold Application]
        F1 --> F2[Anomaly Classification]
        F2 --> F3[Performance Metrics]
        F3 --> F4[Visualizations]
        F3 --> F5[Results Export]
    end
    
    %% Styling
    classDef input fill:#e3f2fd
    classDef graph fill:#f3e5f5
    classDef feature fill:#e8f5e8
    classDef model fill:#fff3e0
    classDef detection fill:#fce4ec
    classDef output fill:#f1f8e9
    
    class A1,A2,A3 input
    class B1,B2,B3,B4,B5 graph
    class C1,C2,C3,C4,C5 feature
    class D1,D2,D3,D4,D5,D6,D7,D8,D9 model
    class E1,E2,E3,E4,E5,E6,E7,E8,E9 detection
    class F1,F2,F3,F4,F5 output
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

## Evaluation Data Generation Pipeline (`generate_eval_data.py`)

### Core Implementation Details

#### 1. Domain-Specific Configuration

The evaluation data generation is specifically designed for **financial services domain**:

```python
OUTPUT_BASE_DIR = 'eval_dataset'
MODEL = 'gpt-4o'
AVAILABLE_TOOLS = ["calculator", "retrieve_knowledge", "final_answer"]
AVAILABLE_AGENTS = ["user", "principal_agent", "home_loan_agent", "credit_agent", "supervisor_agent"]
```

#### 2. Financial Domain Anomaly Descriptors

The system includes domain-specific anomaly descriptions for financial scenarios:

```python
anomaly_descriptors = {
    "Suboptimal Path": "taking suboptimal paths that lead to inefficient thinking or time-consuming steps, such as unnecessary tool calls, redundant actions, or inefficient sequencing in the context of home loan processing, credit checks, or supervision.",
    "Tool Calling Error": "inaccurate usage of a tool or calling the wrong tool for the task, like using calculator for knowledge retrieval.",
    "Agent Handoff Error": "handing off to the wrong agent, such as sending credit check to home_loan_agent instead of credit_agent, leading to mishandling or delays.",
    # ... additional financial-specific descriptions
}
```

#### 3. Parallel Generation with ThreadPoolExecutor

The system uses parallel processing for efficient data generation:

```python
def generate_trace(category, i):
    prompt = get_prompt(category)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates synthetic agent traces with specified characteristics for financial agent systems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=4096
    )
    # ... processing logic

# Parallel execution
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [executor.submit(generate_trace, category, i) for i in range(1, args.num_traces + 1)]
    for future in as_completed(futures):
        trace = future.result()
        # ... save trace
```

#### 4. Dynamic Prompt Generation

The system generates context-aware prompts based on category:

```python
def get_prompt(category):
    if category == 'Normal':
        specific_inst = "a normal successful agent trajectory where the task is fully completed correctly, perhaps with user follow-ups, interruptions, or back-and-forth conversations. There are no errors, and the agent uses appropriate tools and handoffs between principal_agent, home_loan_agent, credit_agent, and supervisor_agent."
        task_completed_str = "true"
        errors_str = 'errors: []'
    else:
        description = anomaly_descriptors.get(category, category.lower())
        specific_inst = f"an anomalous agent trajectory with the specific anomaly type '{category}', which involves {description}."
        task_completed_str = "false"
        errors_str = f'errors: ["... related to {category} ..."]'
    
    return f'''
    You are an expert at generating synthetic agent traces for a reasoning pipeline in a financial chatbot system. Generate 1 synthetic trace in the following JSON schema, where:
    - {specific_inst}
    - The conversations should be nuanced and dynamic, with possible user interruptions, back and forth interactions, to be able to train a dynamic anomaly scorer that can adapt to new and unseen use cases.
    - The scenario should be related to home loans, mortgages, credit checks, financial calculations, or similar financial advice.
    '''
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

## Evaluation Pipeline (`eval_pipeline.py`)

### Core Implementation Details

#### 1. Model Loading and Configuration

The evaluation pipeline loads pre-trained models with proper dimension handling:

```python
def load_model_config():
    """Load model configuration from training artifacts."""
    config_path = os.path.join(MODEL_DIR, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def load_model_artifact(filename):
    """Load model artifacts (scaler, feature mask, etc.)."""
    artifact_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(artifact_path):
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)
    return None
```

#### 2. Enhanced Feature Engineering for Evaluation

The evaluation pipeline includes additional features for better generalization:

```python
def extract_trace_features(trace, embedder=None):
    """Extract numeric and categorical features from trace for anomaly detection."""
    # ... existing feature extraction ...
    
    # Enhanced features for evaluation
    features.update({
        'has_error': len(errors) > 0,
        'error_rate': len(errors) / max(num_steps, 1),
        'efficiency_score': (1 if task_completed else 0) / max(duration_minutes, 1),
        'error_severity': sum([len(e) for e in errors]),
        'completion_efficiency': (1 if task_completed else 0) * num_steps / max(duration_minutes, 1),
        'step_count_anomaly': abs(num_steps - 10) / 10,
        'duration_anomaly': abs(duration_minutes - 5) / 5,
        'error_keyword_count': sum(text.lower().count(word) for text in step_texts for word in ['error', 'fail', 'exception', 'warning']),
    })
    
    return features
```

#### 3. Robust Embedding System

The evaluation pipeline includes fallback embedding mechanisms:

```python
# Load embedder with fallback
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
```

#### 4. Comprehensive Evaluation Metrics

The evaluation pipeline computes detailed performance metrics:

```python
def evaluate_anomaly_detection(y_true, y_scores):
    """Evaluate anomaly detection performance with optimal threshold finding."""
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    # Compute predictions with optimal threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'threshold': best_threshold
    }
```

#### 5. Detailed Results Export

The evaluation pipeline exports comprehensive results:

```python
# Save evaluation results
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

### Domain Adaptation Features

#### 1. Financial Domain Specificity

The evaluation pipeline is specifically designed for financial services:

- **Domain-specific tools**: calculator, retrieve_knowledge, final_answer
- **Financial agents**: home_loan_agent, credit_agent, supervisor_agent
- **Financial anomaly types**: Credit-related errors, loan processing issues
- **Financial scenarios**: Home loans, mortgages, credit checks

#### 2. Cross-Domain Generalization

The system demonstrates generalization capabilities:

- **Unseen anomaly types**: Handles new anomaly patterns
- **Different domains**: Adapts to financial vs. general scenarios
- **Tool variations**: Works with different tool sets
- **Agent variations**: Adapts to different agent configurations

This technical implementation provides a robust, scalable, and efficient anomaly detection system for multi-agent reasoning traces with specific capabilities for financial services evaluation and cross-domain generalization. 