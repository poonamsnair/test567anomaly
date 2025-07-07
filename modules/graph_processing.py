"""
Graph processing and embedding generation for AI Agent Trajectory Anomaly Detection.

This module converts agent trajectories to NetworkX graphs and generates embeddings
using Node2Vec and DeepWalk algorithms with comprehensive hyperparameter tuning.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# Add these imports for GraphSAGE and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SAGEConv
except ImportError:
    torch = None
    nn = None
    optim = None
    Data = None
    DataLoader = None
    SAGEConv = None

from .utils import AgentTrajectory, Timer, ensure_directory

logger = logging.getLogger(__name__)


class GraphProcessor:
    """
    Processes agent trajectories and converts them to graph representations.
    
    This class handles:
    - Converting trajectories to NetworkX DiGraph objects
    - Extracting graph-level features
    - Generating node and graph embeddings using Node2Vec and DeepWalk
    - Hyperparameter tuning for embedding algorithms
    """
    
    def __init__(self, config: Dict):
        """
        Initialize graph processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.graph_config = config.get('graph_processing', {})
        
        # Node and edge feature extractors
        self.node_features = {}
        self.edge_features = {}
        
        logger.info("GraphProcessor initialized")
    
    def trajectories_to_graphs(self, trajectories: List[AgentTrajectory]) -> List[nx.DiGraph]:
        """
        Convert trajectories to NetworkX DiGraph objects.
        
        Args:
            trajectories: List of agent trajectories
        
        Returns:
            List of NetworkX directed graphs
        """
        graphs = []
        
        logger.info("Converting %d trajectories to graphs", len(trajectories))
        
        with Timer() as timer:
            for trajectory in tqdm(trajectories, desc="Converting to graphs"):
                graph = self._trajectory_to_graph(trajectory)
                graphs.append(graph)
        
        logger.info("Converted trajectories to graphs in %.2f seconds", timer.elapsed())
        
        return graphs
    
    def _trajectory_to_graph(self, trajectory: AgentTrajectory) -> nx.DiGraph:
        """Convert a single trajectory to a NetworkX DiGraph."""
        graph = nx.DiGraph()
        
        # Add trajectory-level attributes
        graph.graph['trajectory_id'] = trajectory.trajectory_id
        graph.graph['task_description'] = trajectory.task_description
        graph.graph['is_anomalous'] = trajectory.is_anomalous
        graph.graph['total_duration'] = trajectory.total_duration
        graph.graph['success'] = trajectory.success
        graph.graph['completion_rate'] = trajectory.completion_rate
        
        if trajectory.anomaly_types:
            graph.graph['anomaly_types'] = [at.value for at in trajectory.anomaly_types]
        
        if trajectory.anomaly_severity:
            graph.graph['anomaly_severity'] = trajectory.anomaly_severity.value
        
        # Add nodes with comprehensive attributes
        for node in trajectory.nodes:
            node_attrs = self._extract_node_attributes(node)
            graph.add_node(node.node_id, **node_attrs)
        
        # Add edges with attributes
        for edge in trajectory.edges:
            edge_attrs = self._extract_edge_attributes(edge)
            
            # Ensure both nodes exist
            if edge.source_node_id in graph and edge.target_node_id in graph:
                graph.add_edge(edge.source_node_id, edge.target_node_id, **edge_attrs)
        
        return graph
    
    def _extract_node_attributes(self, node) -> Dict[str, Any]:
        """Extract comprehensive node attributes for graph representation."""
        attrs = {
            # Core attributes
            'node_type': node.node_type.value,
            'agent_type': node.agent_type.value,
            'duration': node.duration or 0.0,
            'status': node.status,
            
            # Performance metrics
            'performance_score': node.get_performance_score(),
            'cpu_usage': node.cpu_usage or 0.0,
            'memory_usage': node.memory_usage or 0.0,
            'network_calls': node.network_calls,
            'retry_count': node.retry_count,
            
            # Success metrics
            'is_failed': node.is_failed(),
            'success': not node.is_failed(),
            
            # Anomaly information
            'is_anomalous': node.is_anomalous,
            'anomaly_type': node.anomaly_type.value if node.anomaly_type else None,
            'anomaly_severity': node.anomaly_severity.value if node.anomaly_severity else None,
            
            # Type-specific attributes
            'tool_type': node.tool_type.value if node.tool_type else None,
            'tool_success': getattr(node, 'tool_success', True),
            'memory_operation': getattr(node, 'memory_operation', None),
            'planning_type': getattr(node, 'planning_type', None),
            'reasoning_type': getattr(node, 'reasoning_type', None),
            'validation_result': getattr(node, 'validation_result', True),
            
            # Confidence scores
            'planning_confidence': getattr(node, 'planning_confidence', None),
            'reasoning_confidence': getattr(node, 'reasoning_confidence', None),
            'validation_score': getattr(node, 'validation_score', None),
            
            # Handoff information
            'source_agent': node.source_agent.value if getattr(node, 'source_agent', None) else None,
            'target_agent': node.target_agent.value if getattr(node, 'target_agent', None) else None,
            'handoff_success': getattr(node, 'handoff_success', True),
        }
        
        # Add start time as timestamp for temporal analysis
        if node.start_time:
            attrs['start_timestamp'] = node.start_time.timestamp()
        
        return attrs
    
    def _extract_edge_attributes(self, edge) -> Dict[str, Any]:
        """Extract comprehensive edge attributes for graph representation."""
        return {
            'edge_type': edge.edge_type,
            'relationship_type': edge.relationship_type,
            'latency': edge.latency or 0.0,
            'probability': edge.probability,
            'confidence': edge.confidence,
            'success_rate': edge.success_rate,
            'error_count': edge.error_count,
            'timeout_count': edge.timeout_count,
            'weight': edge.weight,
            'is_anomalous': edge.is_anomalous,
            'reliability_score': edge.get_reliability_score(),
            'data_size': len(str(edge.data_transferred)) if edge.data_transferred else 0
        }
    
    def generate_node2vec_embeddings(self, graphs: List[nx.DiGraph], 
                                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Generate Node2Vec embeddings with hyperparameter tuning.
        
        Args:
            graphs: List of NetworkX graphs
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary containing embeddings and best parameters
        """
        logger.info("Generating Node2Vec embeddings for %d graphs", len(graphs))
        
        if hyperparameter_tuning:
            return self._tune_node2vec_hyperparameters(graphs)
        else:
            # Use default parameters
            default_params = {
                'dimensions': 128,
                'walk_length': 50,
                'num_walks': 500,
                'p': 1.0,
                'q': 1.0,
                'window': 10,
                'min_count': 1,
                'workers': 4
            }
            
            embeddings = self._generate_node2vec_with_params(graphs, default_params)
            return {
                'embeddings': embeddings,
                'best_params': default_params,
                'method': 'node2vec'
            }
    
    def _tune_node2vec_hyperparameters(self, graphs: List[nx.DiGraph]) -> Dict[str, Any]:
        """Tune Node2Vec hyperparameters using grid search."""
        node2vec_config = self.graph_config.get('node2vec', {})
        
        # Create parameter grid
        param_grid = {
            'dimensions': node2vec_config.get('dimensions', [64, 128, 256]),
            'walk_length': node2vec_config.get('walk_length', [30, 50, 80]),
            'num_walks': node2vec_config.get('num_walks', [200, 500, 1000]),
            'p': node2vec_config.get('p', [0.5, 1.0, 2.0]),
            'q': node2vec_config.get('q', [0.5, 1.0, 2.0]),
            'window': node2vec_config.get('window', [5, 10, 15]),
            'min_count': node2vec_config.get('min_count', [1, 3, 5])
        }
        
        # Sample parameter combinations (limit to avoid excessive computation)
        all_combinations = list(ParameterGrid(param_grid))
        max_combinations = 20  # Limit for practical computation
        
        if len(all_combinations) > max_combinations:
            import random
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info("Tuning Node2Vec with %d parameter combinations", len(combinations))
        
        best_score = -np.inf
        best_params = None
        best_embeddings = None
        
        for params in tqdm(combinations, desc="Tuning Node2Vec"):
            try:
                embeddings = self._generate_node2vec_with_params(graphs, params)
                score = self._evaluate_embeddings(embeddings, graphs)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_embeddings = embeddings
                    
            except Exception as e:
                logger.warning("Failed to generate embeddings with params %s: %s", params, e)
                continue
        
        logger.info("Best Node2Vec score: %.4f with params: %s", best_score, best_params)
        
        return {
            'embeddings': best_embeddings,
            'best_params': best_params,
            'best_score': best_score,
            'method': 'node2vec'
        }
    
    def _generate_node2vec_with_params(self, graphs: List[nx.DiGraph], params: Dict) -> Dict[str, np.ndarray]:
        """Generate Node2Vec embeddings with specific parameters."""
        all_embeddings = {}
        
        for i, graph in enumerate(graphs):
            if len(graph.nodes) == 0:
                # Empty graph - create zero embedding
                all_embeddings[f"graph_{i}"] = np.zeros(params['dimensions'])
                continue
            
            try:
                # Create Node2Vec model
                node2vec = Node2Vec(
                    graph,
                    dimensions=params['dimensions'],
                    walk_length=params['walk_length'],
                    num_walks=params['num_walks'],
                    p=params['p'],
                    q=params['q'],
                    workers=params.get('workers', 4),
                    quiet=True
                )
                
                # Train model
                model = node2vec.fit(
                    window=params['window'],
                    min_count=params['min_count'],
                    batch_words=4
                )
                
                # Get node embeddings
                node_embeddings = {}
                for node_id in graph.nodes():
                    try:
                        node_embeddings[node_id] = model.wv[str(node_id)]
                    except KeyError:
                        # Node not in vocabulary, create zero embedding
                        node_embeddings[node_id] = np.zeros(params['dimensions'])
                
                # Aggregate to graph-level embedding
                graph_embedding = self._aggregate_node_embeddings(node_embeddings, graph, params['dimensions'])
                all_embeddings[f"graph_{i}"] = graph_embedding
                
            except Exception as e:
                logger.warning("Failed to process graph %d: %s", i, e)
                all_embeddings[f"graph_{i}"] = np.zeros(params['dimensions'])
        
        return all_embeddings
    
    def generate_deepwalk_embeddings(self, graphs: List[nx.DiGraph], 
                                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Generate DeepWalk embeddings with hyperparameter tuning.
        
        Args:
            graphs: List of NetworkX graphs
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary containing embeddings and best parameters
        """
        logger.info("Generating DeepWalk embeddings for %d graphs", len(graphs))
        
        if hyperparameter_tuning:
            return self._tune_deepwalk_hyperparameters(graphs)
        else:
            # Use default parameters
            default_params = {
                'dimensions': 128,
                'walk_length': 80,
                'num_walks': 200,
                'window': 10,
                'min_count': 1,
                'sg': 1
            }
            
            embeddings = self._generate_deepwalk_with_params(graphs, default_params)
            return {
                'embeddings': embeddings,
                'best_params': default_params,
                'method': 'deepwalk'
            }
    
    def _tune_deepwalk_hyperparameters(self, graphs: List[nx.DiGraph]) -> Dict[str, Any]:
        """Tune DeepWalk hyperparameters using grid search."""
        deepwalk_config = self.graph_config.get('deepwalk', {})
        
        # Create parameter grid
        param_grid = {
            'dimensions': deepwalk_config.get('dimensions', [64, 128, 256]),
            'walk_length': deepwalk_config.get('walk_length', [40, 80, 120]),
            'num_walks': deepwalk_config.get('num_walks', [80, 200, 400]),
            'window': deepwalk_config.get('window', [5, 10, 15]),
            'min_count': deepwalk_config.get('min_count', [0, 1, 3])
        }
        
        # Sample parameter combinations
        all_combinations = list(ParameterGrid(param_grid))
        max_combinations = 15  # Limit for practical computation
        
        if len(all_combinations) > max_combinations:
            import random
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info("Tuning DeepWalk with %d parameter combinations", len(combinations))
        
        best_score = -np.inf
        best_params = None
        best_embeddings = None
        
        for params in tqdm(combinations, desc="Tuning DeepWalk"):
            try:
                embeddings = self._generate_deepwalk_with_params(graphs, params)
                score = self._evaluate_embeddings(embeddings, graphs)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_embeddings = embeddings
                    
            except Exception as e:
                logger.warning("Failed to generate embeddings with params %s: %s", params, e)
                continue
        
        logger.info("Best DeepWalk score: %.4f with params: %s", best_score, best_params)
        
        return {
            'embeddings': best_embeddings,
            'best_params': best_params,
            'best_score': best_score,
            'method': 'deepwalk'
        }
    
    def _generate_deepwalk_with_params(self, graphs: List[nx.DiGraph], params: Dict) -> Dict[str, np.ndarray]:
        """Generate DeepWalk embeddings with specific parameters."""
        all_embeddings = {}
        
        for i, graph in enumerate(graphs):
            if len(graph.nodes) == 0:
                # Empty graph - create zero embedding
                all_embeddings[f"graph_{i}"] = np.zeros(params['dimensions'])
                continue
            
            try:
                # Generate random walks
                walks = self._generate_random_walks(
                    graph, 
                    num_walks=params['num_walks'],
                    walk_length=params['walk_length']
                )
                
                if not walks:
                    all_embeddings[f"graph_{i}"] = np.zeros(params['dimensions'])
                    continue
                
                # Train Word2Vec model on walks
                model = Word2Vec(
                    walks,
                    vector_size=params['dimensions'],
                    window=params['window'],
                    min_count=params['min_count'],
                    sg=params.get('sg', 1),  # Skip-gram
                    workers=1,  # Single worker for reproducibility
                    epochs=10
                )
                
                # Get node embeddings
                node_embeddings = {}
                for node_id in graph.nodes():
                    try:
                        node_embeddings[node_id] = model.wv[str(node_id)]
                    except KeyError:
                        # Node not in vocabulary, create zero embedding
                        node_embeddings[node_id] = np.zeros(params['dimensions'])
                
                # Aggregate to graph-level embedding
                graph_embedding = self._aggregate_node_embeddings(node_embeddings, graph, params['dimensions'])
                all_embeddings[f"graph_{i}"] = graph_embedding
                
            except Exception as e:
                logger.warning("Failed to process graph %d: %s", i, e)
                all_embeddings[f"graph_{i}"] = np.zeros(params['dimensions'])
        
        return all_embeddings
    
    def _generate_random_walks(self, graph: nx.DiGraph, num_walks: int, walk_length: int) -> List[List[str]]:
        """Generate random walks for DeepWalk."""
        walks = []
        nodes = list(graph.nodes())
        
        for _ in range(num_walks):
            if not nodes:
                break
                
            # Start walk from random node
            start_node = np.random.choice(nodes)
            walk = [str(start_node)]
            current_node = start_node
            
            for _ in range(walk_length - 1):
                neighbors = list(graph.successors(current_node))
                if not neighbors:
                    break
                
                # Choose next node based on edge weights
                weights = []
                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    weights.append(weight)
                
                # Normalize weights
                if sum(weights) > 0:
                    weights = np.array(weights) / sum(weights)
                    next_node = np.random.choice(neighbors, p=weights)
                else:
                    next_node = np.random.choice(neighbors)
                
                walk.append(str(next_node))
                current_node = next_node
            
            walks.append(walk)
        
        return walks
    
    def _aggregate_node_embeddings(self, node_embeddings: Dict[str, np.ndarray], 
                                 graph: nx.DiGraph, dimensions: int) -> np.ndarray:
        """Aggregate node embeddings to graph-level embedding."""
        aggregation_methods = self.graph_config.get('aggregation_methods', ['mean'])
        
        if not node_embeddings:
            return np.zeros(dimensions)
        
        embeddings_matrix = np.array(list(node_embeddings.values()))
        
        # Calculate centrality measures for weighted aggregation
        try:
            centralities = self._calculate_centralities(graph)
        except:
            centralities = {}
        
        aggregated_features = []
        
        for method in aggregation_methods:
            if method == 'mean':
                agg = np.mean(embeddings_matrix, axis=0)
            elif method == 'max':
                agg = np.max(embeddings_matrix, axis=0)
            elif method == 'sum':
                agg = np.sum(embeddings_matrix, axis=0)
            elif method == 'weighted_mean' and centralities:
                # Weight by node importance (centrality)
                weights = []
                for node_id in node_embeddings.keys():
                    weight = centralities.get(node_id, 1.0)
                    weights.append(weight)
                
                weights = np.array(weights)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                    agg = np.average(embeddings_matrix, axis=0, weights=weights)
                else:
                    agg = np.mean(embeddings_matrix, axis=0)
            else:
                agg = np.mean(embeddings_matrix, axis=0)  # Default to mean
            
            aggregated_features.extend(agg)
        
        return np.array(aggregated_features)
    
    def _calculate_centralities(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate various centrality measures for nodes."""
        centralities = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(graph)
        
        # Betweenness centrality
        try:
            betweenness_cent = nx.betweenness_centrality(graph)
        except:
            betweenness_cent = {}
        
        # Closeness centrality
        try:
            closeness_cent = nx.closeness_centrality(graph)
        except:
            closeness_cent = {}
        
        # Eigenvector centrality
        try:
            eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            eigenvector_cent = {}
        
        # Combine centralities
        for node in graph.nodes():
            centrality_score = (
                degree_cent.get(node, 0) +
                betweenness_cent.get(node, 0) +
                closeness_cent.get(node, 0) +
                eigenvector_cent.get(node, 0)
            ) / 4.0
            
            centralities[node] = centrality_score
        
        return centralities
    
    def _evaluate_embeddings(self, embeddings: Dict[str, np.ndarray], graphs: List[nx.DiGraph]) -> float:
        """Evaluate embedding quality using graph structure preservation."""
        if not embeddings or len(embeddings) < 2:
            return 0.0
        
        try:
            # Calculate embedding similarity matrix
            embedding_matrix = np.array(list(embeddings.values()))
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            embedding_sim = cosine_similarity(embedding_matrix)
            
            # Calculate graph structural similarity (simplified)
            graph_sim = np.zeros((len(graphs), len(graphs)))
            
            for i, graph_i in enumerate(graphs):
                for j, graph_j in enumerate(graphs):
                    if i != j:
                        # Simple structural similarity based on node and edge counts
                        nodes_i, edges_i = len(graph_i.nodes), len(graph_i.edges)
                        nodes_j, edges_j = len(graph_j.nodes), len(graph_j.edges)
                        
                        if nodes_i + nodes_j > 0 and edges_i + edges_j > 0:
                            node_sim = 1.0 - abs(nodes_i - nodes_j) / max(nodes_i, nodes_j, 1)
                            edge_sim = 1.0 - abs(edges_i - edges_j) / max(edges_i, edges_j, 1)
                            graph_sim[i][j] = (node_sim + edge_sim) / 2.0
            
            # Calculate correlation between similarities
            from scipy.stats import pearsonr
            
            # Flatten similarity matrices and calculate correlation
            embedding_sim_flat = embedding_sim[np.triu_indices_from(embedding_sim, k=1)]
            graph_sim_flat = graph_sim[np.triu_indices_from(graph_sim, k=1)]
            
            if len(embedding_sim_flat) > 1 and np.std(embedding_sim_flat) > 0 and np.std(graph_sim_flat) > 0:
                correlation, _ = pearsonr(embedding_sim_flat, graph_sim_flat)
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning("Failed to evaluate embeddings: %s", e)
            return 0.0
    
    def save_embeddings(self, embeddings_data: Dict[str, Any], filepath: str) -> None:
        """Save embeddings to file."""
        ensure_directory(filepath).parent
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        logger.info("Saved embeddings to %s", filepath)
    
    def load_embeddings(self, filepath: str) -> Dict[str, Any]:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info("Loaded embeddings from %s", filepath)
        return embeddings
    
    def load_graphs(self, filepath: str) -> List[nx.DiGraph]:
        """Load graphs from file."""
        with open(filepath, 'rb') as f:
            graphs = pickle.load(f)
        
        logger.info("Loaded %d graphs from %s", len(graphs), filepath)
        return graphs
    
    def get_graph_statistics(self, graphs: List[nx.DiGraph]) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph dataset."""
        stats = {
            'total_graphs': len(graphs),
            'node_count': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'edge_count': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'density': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'node_types': {},
            'edge_types': {},
            'anomalous_graphs': 0,
            'avg_path_length': 0
        }
        
        if not graphs:
            return stats
        
        node_counts = []
        edge_counts = []
        densities = []
        path_lengths = []
        
        for graph in graphs:
            # Basic counts
            num_nodes = len(graph.nodes)
            num_edges = len(graph.edges)
            
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
            
            # Density
            if num_nodes > 1:
                density = nx.density(graph)
                densities.append(density)
            
            # Average path length
            if num_nodes > 1:
                try:
                    if nx.is_weakly_connected(graph):
                        avg_path = nx.average_shortest_path_length(graph.to_undirected())
                        path_lengths.append(avg_path)
                except:
                    pass
            
            # Count node types
            for node_id, node_data in graph.nodes(data=True):
                node_type = node_data.get('node_type', 'unknown')
                stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            # Count edge types
            for _, _, edge_data in graph.edges(data=True):
                edge_type = edge_data.get('edge_type', 'unknown')
                stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
            
            # Count anomalous graphs
            if graph.graph.get('is_anomalous', False):
                stats['anomalous_graphs'] += 1
        
        # Calculate statistics
        if node_counts:
            stats['node_count'] = {
                'min': min(node_counts),
                'max': max(node_counts),
                'mean': np.mean(node_counts),
                'std': np.std(node_counts)
            }
        
        if edge_counts:
            stats['edge_count'] = {
                'min': min(edge_counts),
                'max': max(edge_counts),
                'mean': np.mean(edge_counts),
                'std': np.std(edge_counts)
            }
        
        if densities:
            stats['density'] = {
                'min': min(densities),
                'max': max(densities),
                'mean': np.mean(densities),
                'std': np.std(densities)
            }
        
        if path_lengths:
            stats['avg_path_length'] = np.mean(path_lengths)
        
        return stats
    
    def generate_graphsage_embeddings(self, graphs: List[nx.DiGraph], 
                                    hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Generate GraphSAGE embeddings with hyperparameter tuning.
        
        Args:
            graphs: List of NetworkX graphs
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary containing embeddings and best parameters
        """
        logger.info("Generating GraphSAGE embeddings for %d graphs", len(graphs))
        
        # Check if PyTorch Geometric is available
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch_geometric.data import Data, DataLoader
            from torch_geometric.nn import SAGEConv
            TORCH_GEOMETRIC_AVAILABLE = True
        except ImportError:
            logger.warning("PyTorch Geometric not available. Skipping GraphSAGE.")
            return {
                'embeddings': {},
                'best_params': {},
                'method': 'graphsage',
                'error': 'PyTorch Geometric not available'
            }
        
        if hyperparameter_tuning:
            return self._tune_graphsage_hyperparameters(graphs)
        else:
            # Use default parameters
            default_params = {
                'hidden_dims': [64, 128],
                'output_dim': 64,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'dropout': 0.1,
                'aggregator': 'mean'
            }
            
            embeddings = self._generate_graphsage_with_params(graphs, default_params)
            return {
                'embeddings': embeddings,
                'best_params': default_params,
                'method': 'graphsage'
            }
    
    def _tune_graphsage_hyperparameters(self, graphs: List[nx.DiGraph]) -> Dict[str, Any]:
        """Tune GraphSAGE hyperparameters using grid search."""
        graphsage_config = self.graph_config.get('graphsage', {})
        
        # Create parameter grid
        param_grid = {
            'hidden_dims': graphsage_config.get('hidden_dims', [[32, 64], [64, 128]]),
            'output_dim': graphsage_config.get('output_dim', [32, 64]),
            'learning_rate': graphsage_config.get('learning_rate', [0.001, 0.01]),
            'epochs': [50],  # Reduced for tuning
            'batch_size': graphsage_config.get('batch_size', [16, 32]),
            'dropout': graphsage_config.get('dropout', [0.1, 0.3]),
            'aggregator': graphsage_config.get('aggregator', ['mean', 'max'])
        }
        
        # Sample parameter combinations
        all_combinations = list(ParameterGrid(param_grid))
        max_combinations = 10  # Limit for practical computation
        
        if len(all_combinations) > max_combinations:
            import random
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        logger.info("Tuning GraphSAGE with %d parameter combinations", len(combinations))
        
        best_score = -np.inf
        best_params = None
        best_embeddings = None
        
        for params in tqdm(combinations, desc="Tuning GraphSAGE"):
            try:
                embeddings = self._generate_graphsage_with_params(graphs, params)
                score = self._evaluate_embeddings(embeddings, graphs)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_embeddings = embeddings
                    
            except Exception as e:
                logger.warning("Failed to generate GraphSAGE embeddings with params %s: %s", params, e)
                continue
        
        logger.info("Best GraphSAGE score: %.4f with params: %s", best_score, best_params)
        
        return {
            'embeddings': best_embeddings,
            'best_params': best_params,
            'best_score': best_score,
            'method': 'graphsage'
        }
    
    def _generate_graphsage_with_params(self, graphs: List[nx.DiGraph], params: Dict) -> Dict[str, np.ndarray]:
        """Generate GraphSAGE embeddings with specific parameters."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch_geometric.data import Data, DataLoader
            from torch_geometric.nn import SAGEConv
        except ImportError:
            logger.warning("PyTorch Geometric not available for GraphSAGE")
            return {}
        
        all_embeddings = {}
        
        # Convert graphs to PyTorch Geometric format
        torch_graphs = self._convert_to_torch_geometric(graphs)
        
        if not torch_graphs:
            logger.warning("No valid graphs for GraphSAGE training")
            return {}
        
        # Create and train GraphSAGE model
        model = self._create_graphsage_model(torch_graphs[0], params)
        trained_model = self._train_graphsage_model(model, torch_graphs, params)
        
        # Generate embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(torch_graphs):
                try:
                    data = data.to(device)
                    node_embeddings, _ = trained_model(data)
                    
                    # Aggregate to graph-level embedding
                    if hasattr(data, 'num_graphs') and data.num_graphs > 1:
                        # Handle batched graphs
                        graph_embeddings = []
                        for j in range(data.num_graphs):
                            mask = data.batch == j
                            graph_emb = torch.mean(node_embeddings[mask], dim=0)
                            graph_embeddings.append(graph_emb)
                        graph_embedding = torch.stack(graph_embeddings).mean(dim=0)
                    else:
                        # Single graph
                        graph_embedding = torch.mean(node_embeddings, dim=0)
                    
                    all_embeddings[f"graph_{i}"] = graph_embedding.cpu().numpy()
                    
                except Exception as e:
                    logger.warning("Failed to generate embedding for graph %d: %s", i, e)
                    all_embeddings[f"graph_{i}"] = np.zeros(params['output_dim'])
        
        return all_embeddings
    
    def _create_graphsage_model(self, sample_data, params: Dict):
        """Create GraphSAGE model with consistent input dimension handling."""
        # Determine input dimension from node features
        if hasattr(sample_data, 'x') and sample_data.x is not None:
            input_dim = sample_data.x.shape[1]
        else:
            input_dim = 15  # Default based on node features
        
        # Store input dimension in params for consistency
        params['input_dim'] = input_dim
        
        hidden_dims = params['hidden_dims']
        output_dim = params['output_dim']
        dropout = params['dropout']
        
        # Adjust hidden dimensions if they're too large compared to input
        if input_dim < 20:
            adjusted_hidden_dims = []
            for dim in hidden_dims:
                if dim > input_dim * 2:
                    adjusted_dim = max(input_dim, dim // 2)
                    logger.warning("GraphSAGE hidden dimension %d is very large compared to input %d, reducing to %d", 
                                  dim, input_dim, adjusted_dim)
                    adjusted_hidden_dims.append(adjusted_dim)
                else:
                    adjusted_hidden_dims.append(dim)
            hidden_dims = adjusted_hidden_dims
        
        return GraphSAGEModel(input_dim, hidden_dims, output_dim, dropout)
    
    def _train_graphsage_model(self, model, torch_graphs, params: Dict):
        """Train the GraphSAGE model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create data loader
        loader = DataLoader(torch_graphs, batch_size=params['batch_size'], shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        model.train()
        
        for epoch in range(params['epochs']):
            epoch_loss = 0.0
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                node_embeddings, reconstructed = model(batch)
                
                # Reconstruction loss (autoencoder-style)
                loss = nn.MSELoss()(reconstructed, batch.x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info("GraphSAGE Epoch %d/%d, Loss: %.6f", epoch, params['epochs'], epoch_loss / len(loader))
        
        return model

    def _convert_to_torch_geometric(self, graphs: List[Any]) -> List[Any]:
        """Convert NetworkX graphs to PyTorch Geometric Data objects with consistent dimensions."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            logger.warning("PyTorch Geometric not available for conversion.")
            return []
        
        torch_graphs = []
        expected_dim = None
        
        for graph in graphs:
            try:
                if len(graph.nodes) == 0:
                    continue
                if len(graph.nodes) < 2:
                    logger.debug("Skipping graph with only %d nodes", len(graph.nodes))
                    continue
                    
                node_features = []
                node_mapping = {node: i for i, node in enumerate(graph.nodes())}
                
                for node in graph.nodes():
                    node_data = graph.nodes[node]
                    features = [
                        float(node_data.get('duration', 0.0)),
                        float(node_data.get('performance_score', 0.0)),
                        float(node_data.get('cpu_usage', 0.0)),
                        float(node_data.get('memory_usage', 0.0)),
                        float(node_data.get('network_calls', 0)),
                        float(node_data.get('retry_count', 0)),
                        1.0 if node_data.get('success', True) else 0.0,
                    ]
                    node_type = node_data.get('node_type', 'unknown')
                    node_types = ['tool_call', 'reasoning', 'memory_access', 'planning', 
                                'validation', 'handoff', 'observation', 'action']
                    for nt in node_types:
                        features.append(1.0 if node_type == nt else 0.0)
                    node_features.append(features)
                
                # Ensure consistent feature dimension
                if expected_dim is None:
                    expected_dim = len(node_features[0])
                
                for i in range(len(node_features)):
                    feat = node_features[i]
                    if len(feat) > expected_dim:
                        # Truncate to expected dimension
                        node_features[i] = feat[:expected_dim]
                    elif len(feat) < expected_dim:
                        # Pad with zeros to expected dimension
                        padding = [0.0] * (expected_dim - len(feat))
                        node_features[i] = feat + padding
                
                edge_indices = []
                edge_attrs = []
                for u, v, edge_data in graph.edges(data=True):
                    if u in node_mapping and v in node_mapping:
                        edge_indices.append([node_mapping[u], node_mapping[v]])
                        edge_attr = [
                            float(edge_data.get('latency', 0.0)),
                            float(edge_data.get('probability', 1.0)),
                            float(edge_data.get('confidence', 1.0)),
                            float(edge_data.get('success_rate', 1.0)),
                            float(edge_data.get('weight', 1.0))
                        ]
                        edge_attrs.append(edge_attr)
                
                if not edge_indices:
                    edge_indices = [[i, i] for i in range(len(node_features))]
                    edge_attrs = [[0.0, 1.0, 1.0, 1.0, 1.0] for _ in range(len(node_features))]
                
                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                torch_graphs.append(data)
                
            except Exception as e:
                logger.warning(f"Failed to convert graph: {e}")
                continue
        
        logger.info(f"Converted {len(torch_graphs)} graphs to PyTorch Geometric format (GraphProcessor)")
        return torch_graphs


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for node embedding generation.
    
    This model uses GraphSAGE layers to learn node representations
    that capture both local structure and node features.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(GraphSAGEModel, self).__init__()
        
        self.dropout = dropout
        
        # GraphSAGE layers
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = SAGEConv(prev_dim, hidden_dim)
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = SAGEConv(prev_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final projection layer to input_dim for reconstruction
        self.reconstruction_layer = nn.Linear(output_dim, input_dim)
    
    def forward(self, data):
        """Forward pass through the GraphSAGE model."""
        x, edge_index = data.x, data.edge_index
        
        # GraphSAGE layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        # Output layer
        node_embeddings = self.output_layer(x, edge_index)
        
        # Project to input_dim for reconstruction
        reconstructed = self.reconstruction_layer(node_embeddings)
        
        return node_embeddings, reconstructed
