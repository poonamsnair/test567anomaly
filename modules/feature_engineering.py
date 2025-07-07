"""
Feature engineering for AI Agent Trajectory Anomaly Detection.

This module extracts comprehensive features from trajectory graphs including:
- Structural features (topology, connectivity, centrality)
- DAG-specific features (topological properties, branching)
- Temporal features (duration analysis, execution patterns)
- Semantic features (agent behavior, tool usage, error patterns)
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .utils import Timer

logger = logging.getLogger(__name__)

# Configuration constants to replace hardcoded values
class FeatureConfig:
    """Configuration constants for feature engineering."""
    # Infinite value replacement
    INF_REPLACEMENT_POSITIVE = 1e6
    INF_REPLACEMENT_NEGATIVE = -1e6
    
    # Missing value thresholds
    MISSING_VALUE_THRESHOLD = 0.5  # If >50% missing, use median imputation
    
    # Default values for disconnected graphs
    DEFAULT_DIAMETER_MULTIPLIER = 1.0  # Use num_nodes - 1 for disconnected graphs
    
    # Centrality calculation parameters
    CENTRALITY_TYPES = ['betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']
    
    # Edge attribute statistics
    LATENCY_PERCENTILES = [95, 99]
    
    # Feature categories for analysis
    FEATURE_CATEGORIES = {
        'structural': ['num_nodes', 'num_edges', 'density', 'diameter', 'centrality'],
        'temporal': ['duration', 'time', 'latency', 'delay', 'gap'],
        'semantic': ['agent', 'tool', 'error', 'recovery', 'handoff'],
        'dag': ['dag', 'topological', 'branch', 'level', 'parallel']
    }


class FeatureExtractor:
    """
    Extracts comprehensive features from agent trajectory graphs.
    
    This class implements multiple categories of features:
    1. Structural features - Graph topology and connectivity measures
    2. DAG-specific features - Properties specific to directed acyclic graphs
    3. Temporal features - Time-based patterns and execution characteristics
    4. Semantic features - Agent behavior and domain-specific patterns
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        
        # Feature categories to extract
        self.structural_features = self.feature_config.get('structural_features', [])
        self.dag_features = self.feature_config.get('dag_features', [])
        self.temporal_features = self.feature_config.get('temporal_features', [])
        self.semantic_features = self.feature_config.get('semantic_features', [])
        
        logger.info("FeatureExtractor initialized")
    
    def extract_features(self, graphs: List[nx.DiGraph]) -> pd.DataFrame:
        """Extract features from a list of graphs and return as a DataFrame."""
        logger.info("Extracting features from %d graphs", len(graphs))
        features = []
        for idx, graph in enumerate(tqdm(graphs, desc="Extracting features")):
            graph_features = self._extract_graph_features(graph, idx)
            features.append(graph_features)
        features_df = pd.DataFrame(features)
        
        # Simplified numeric conversion and NaN/inf handling
        logger.info("Processing extracted features...")
        features_df = self._clean_and_validate_features(features_df)
        
        logger.info("Feature matrix shape: %s", features_df.shape)
        self._log_feature_statistics(features_df)
        
        return features_df
    
    def _clean_and_validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate feature DataFrame with simplified logic."""
        # Convert to numeric, preserving non-numeric columns
        numeric_columns = []
        for col in features_df.columns:
            if col in ['graph_id', 'is_anomalous', 'success']:
                continue
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                numeric_columns.append(col)
            except Exception:
                logger.warning(f"Could not convert column {col} to numeric")
        
        if not numeric_columns:
            return features_df
        
        # Single-pass cleaning for NaN and Inf values
        numeric_data = features_df[numeric_columns]
        
        # Replace infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], 
                                          [FeatureConfig.INF_REPLACEMENT_POSITIVE, 
                                           FeatureConfig.INF_REPLACEMENT_NEGATIVE])
        
        # Handle NaN values with median imputation
        for col in numeric_columns:
            nan_count = numeric_data[col].isna().sum()
            if nan_count > 0:
                if nan_count > len(numeric_data) * FeatureConfig.MISSING_VALUE_THRESHOLD:
                    # High missing rate - use median
                    median_val = numeric_data[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    numeric_data[col] = numeric_data[col].fillna(median_val)
                    logger.info(f"Column {col}: Used median imputation ({median_val:.4f})")
                else:
                    # Low missing rate - use median for remaining NaN
                    median_val = numeric_data[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    numeric_data[col] = numeric_data[col].fillna(median_val)
        
        # Update DataFrame and defragment
        features_df[numeric_columns] = numeric_data
        features_df = features_df.copy()
        
        return features_df
    
    def _log_feature_statistics(self, features_df: pd.DataFrame) -> None:
        """Log feature statistics for the first 10 numeric columns."""
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        logger.info("Feature statistics:")
        for col in numeric_columns[:10]:
            try:
                col_data = features_df[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    logger.info(f"  {col}: min={col_data.min():.4f}, max={col_data.max():.4f}, "
                              f"mean={col_data.mean():.4f}, std={col_data.std():.4f}")
                else:
                    logger.info(f"  {col}: non-numeric column")
            except Exception as e:
                logger.warning(f"  {col}: could not compute statistics - {e}")
    
    def _extract_graph_features(self, graph: nx.DiGraph, graph_idx: int) -> Dict[str, Any]:
        """Extract all features from a single graph."""
        features = {'graph_id': f"graph_{graph_idx}"}
        
        # Add graph metadata
        features.update(self._extract_metadata_features(graph))
        
        # Extract different categories of features
        if self.structural_features:
            features.update(self._extract_structural_features(graph))
        
        if self.dag_features:
            features.update(self._extract_dag_features(graph))
        
        if self.temporal_features:
            features.update(self._extract_temporal_features(graph))
        
        if self.semantic_features:
            features.update(self._extract_semantic_features(graph))
        
        # Always extract graph structure features (essential for GNNs)
        features.update(self._extract_graph_structure_features(graph))
        
        return features
    
    def _extract_metadata_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract basic metadata features."""
        return {
            'is_anomalous': graph.graph.get('is_anomalous', False),
            'success': graph.graph.get('success', True),
            'completion_rate': graph.graph.get('completion_rate', 1.0),
            'total_duration': graph.graph.get('total_duration', 0.0),
            'anomaly_severity': graph.graph.get('anomaly_severity', None)
        }
    
    def _extract_structural_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract structural features from the graph."""
        features = {}
        
        # Basic counts
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        
        features['num_nodes'] = num_nodes
        features['num_edges'] = num_edges
        
        if num_nodes == 0:
            return self._get_empty_features(self.structural_features)
        
        # Extract features with consolidated error handling
        features.update(self._extract_basic_structural_features(graph, num_nodes))
        features.update(self._extract_centrality_features(graph))
        features.update(self._extract_basic_connectivity_features(graph))
        
        return features
    
    def _extract_basic_structural_features(self, graph: nx.DiGraph, num_nodes: int) -> Dict[str, Any]:
        """Extract basic structural features with consolidated error handling."""
        features = {}
        
        # Density
        if 'density' in self.structural_features:
            features['density'] = self._safe_calculation(lambda: nx.density(graph), 0.0)
        
        # Diameter and path lengths
        if any(feat in self.structural_features for feat in ['diameter', 'average_shortest_path_length']):
            features.update(self._extract_path_length_features(graph, num_nodes))
        
        # Clustering coefficient
        if 'clustering_coefficient' in self.structural_features:
            features['clustering_coefficient'] = self._safe_calculation(
                lambda: np.mean(list(nx.clustering(graph.to_undirected()).values())), 0.0)
        
        # Transitivity
        if 'transitivity' in self.structural_features:
            features['transitivity'] = self._safe_calculation(
                lambda: nx.transitivity(graph.to_undirected()), 0.0)
        
        # Longest path length
        if 'longest_path_length' in self.structural_features:
            features['longest_path_length'] = self._safe_calculation(
                lambda: self._calculate_longest_path(graph), num_nodes - 1)
        
        return features
    
    def _extract_path_length_features(self, graph: nx.DiGraph, num_nodes: int) -> Dict[str, Any]:
        """Extract path length features with proper handling of disconnected graphs."""
        features = {}
        default_value = int(num_nodes * FeatureConfig.DEFAULT_DIAMETER_MULTIPLIER) - 1
        
        try:
            if nx.is_weakly_connected(graph) and num_nodes > 1:
                undirected = graph.to_undirected()
                
                if 'diameter' in self.structural_features:
                    diameter = nx.diameter(undirected)
                    features['diameter'] = diameter if not np.isinf(diameter) else default_value
                
                if 'average_shortest_path_length' in self.structural_features:
                    avg_path = nx.average_shortest_path_length(undirected)
                    features['average_shortest_path_length'] = avg_path if not np.isinf(avg_path) else default_value
            else:
                # For disconnected graphs, use reasonable defaults
                if 'diameter' in self.structural_features:
                    features['diameter'] = default_value
                if 'average_shortest_path_length' in self.structural_features:
                    features['average_shortest_path_length'] = default_value
        except Exception as e:
            logger.warning(f"Error calculating path length features: {e}")
            if 'diameter' in self.structural_features:
                features['diameter'] = default_value
            if 'average_shortest_path_length' in self.structural_features:
                features['average_shortest_path_length'] = default_value
        
        return features
    
    def _extract_centrality_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract centrality features with consolidated logic."""
        features = {}
        
        for centrality_type in FeatureConfig.CENTRALITY_TYPES:
            if centrality_type in self.structural_features:
                try:
                    centrality_values = self._calculate_centrality(graph, centrality_type)
                    if centrality_values and len(centrality_values) > 0:
                        values_list = list(centrality_values.values())
                        valid_values = [v for v in values_list if not (np.isnan(v) or np.isinf(v))]
                        if valid_values:
                            features[f'{centrality_type}_mean'] = np.mean(valid_values)
                            features[f'{centrality_type}_std'] = np.std(valid_values)
                            features[f'{centrality_type}_max'] = np.max(valid_values)
                        else:
                            features.update(self._get_zero_centrality_features(centrality_type))
                    else:
                        features.update(self._get_zero_centrality_features(centrality_type))
                except Exception as e:
                    logger.warning(f"Error calculating {centrality_type}: {e}")
                    features.update(self._get_zero_centrality_features(centrality_type))
        
        return features
    
    def _get_zero_centrality_features(self, centrality_type: str) -> Dict[str, float]:
        """Get zero values for centrality features."""
        return {
            f'{centrality_type}_mean': 0.0,
            f'{centrality_type}_std': 0.0,
            f'{centrality_type}_max': 0.0
        }
    
    def _extract_basic_connectivity_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract basic connectivity features with consolidated error handling."""
        features = {}
        num_nodes = len(graph.nodes)
        
        # Connected components
        if 'number_connected_components' in self.structural_features:
            features['number_connected_components'] = self._safe_calculation(
                lambda: nx.number_weakly_connected_components(graph), 1)
        
        # Node connectivity
        if 'node_connectivity' in self.structural_features:
            if num_nodes > 1:
                features['node_connectivity'] = self._safe_calculation(
                    lambda: nx.node_connectivity(graph), 0)
            else:
                features['node_connectivity'] = 0
        
        return features
    
    def _safe_calculation(self, calculation_func, default_value: Any) -> Any:
        """Safely execute a calculation with error handling."""
        try:
            result = calculation_func()
            if isinstance(result, (int, float)) and (np.isnan(result) or np.isinf(result)):
                return default_value
            return result
        except Exception as e:
            logger.warning(f"Calculation failed: {e}")
            return default_value
    
    def _get_empty_features(self, feature_list: List[str]) -> Dict[str, float]:
        """Get empty feature values for a given feature list."""
        return {feat: 0.0 for feat in feature_list}
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and distributions."""
        analysis = {
            'feature_count': len(features_df.columns),
            'feature_categories': {},
            'missing_values': {},
            'feature_correlations': {},
            'feature_distributions': {}
        }
        
        # Categorize features using configuration constants
        for category, keywords in FeatureConfig.FEATURE_CATEGORIES.items():
            category_features = []
            for col in features_df.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    category_features.append(col)
            analysis['feature_categories'][category] = category_features
        
        # Missing values analysis
        for col in features_df.columns.unique():
            missing_count = features_df[col].isnull().sum()
            if isinstance(missing_count, pd.Series):
                missing_count = missing_count.sum()
            
            if int(missing_count) > 0:
                analysis['missing_values'][col] = int(missing_count)
        
        # Feature distributions (for numeric features)
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature != 'graph_id':
                values = features_df[feature].values
                # Filter out None values and convert to numeric
                clean_values = []
                for v in values:
                    if np.isscalar(v) and v is not None:
                        if isinstance(v, (int, float)):
                            if not (np.isnan(v) or np.isinf(v)):
                                clean_values.append(float(v))
                        else:
                            try:
                                fv = float(v)
                                if not (np.isnan(fv) or np.isinf(fv)):
                                    clean_values.append(fv)
                            except Exception:
                                continue
                
                if clean_values:
                    analysis['feature_distributions'][feature] = {
                        'mean': np.mean(clean_values),
                        'std': np.std(clean_values),
                        'min': np.min(clean_values),
                        'max': np.max(clean_values),
                        'unique_count': len(np.unique(clean_values))
                    }
                else:
                    analysis['feature_distributions'][feature] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'unique_count': 0
                    }
        
        return analysis
    
    def _extract_dag_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract DAG-specific features."""
        features = {}
        
        if len(graph.nodes) == 0:
            return {key: 0.0 for key in self.dag_features}
        
        # Check if graph is actually a DAG
        is_dag = nx.is_directed_acyclic_graph(graph)
        features['is_dag'] = is_dag
        
        # Topological sort validation
        if 'topological_sort_validation' in self.dag_features:
            if is_dag and len(graph) > 0:
                try:
                    list(nx.topological_sort(graph))
                    features['topological_sort_validation'] = True
                except (nx.NetworkXError, nx.NetworkXUnfeasible):
                    features['topological_sort_validation'] = False
            else:
                features['topological_sort_validation'] = False
        
        # Longest path in DAG
        if 'longest_path_in_dag' in self.dag_features and is_dag:
            features['longest_path_in_dag'] = self._dag_longest_path(graph)
        else:
            features['longest_path_in_dag'] = 0
        
        # DAG depth (number of levels)
        if 'dag_depth' in self.dag_features:
            features['dag_depth'] = self._calculate_dag_depth(graph)
        
        # Nodes per level statistics
        if any(feat in self.dag_features for feat in ['nodes_per_level', 'level_width_variance']):
            level_stats = self._calculate_level_statistics(graph)
            features.update(level_stats)
        
        # Branching factor
        if 'branching_factor' in self.dag_features:
            features['branching_factor'] = self._calculate_branching_factor(graph)
        
        # Merge points (nodes with multiple incoming edges)
        if 'merge_points' in self.dag_features:
            merge_points = sum(1 for node in graph.nodes() if graph.in_degree(node) > 1)
            features['merge_points'] = merge_points
        
        # Parallel paths
        if 'parallel_paths' in self.dag_features:
            features['parallel_paths'] = self._count_parallel_paths(graph)
        
        return features
    
    def _extract_temporal_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract temporal features from the graph."""
        features = {}
        
        if len(graph.nodes) == 0:
            return {key: 0.0 for key in self.temporal_features}
        
        # Collect temporal data
        durations = []
        start_times = []
        
        for node_id, node_data in graph.nodes(data=True):
            duration = node_data.get('duration', 0.0)
            start_time = node_data.get('start_timestamp', 0.0)
            
            if duration > 0:
                durations.append(duration)
            if start_time > 0:
                start_times.append(start_time)
        
        # Total duration
        if 'total_duration' in self.temporal_features:
            features['total_duration'] = graph.graph.get('total_duration', sum(durations))
        
        # Duration statistics
        if durations:
            if 'average_node_duration' in self.temporal_features:
                features['average_node_duration'] = np.mean(durations)
            
            if 'duration_variance' in self.temporal_features:
                features['duration_variance'] = np.var(durations)
            
            features['duration_min'] = np.min(durations)
            features['duration_max'] = np.max(durations)
            features['duration_median'] = np.median(durations)
        else:
            features['average_node_duration'] = 0.0
            features['duration_variance'] = 0.0
            features['duration_min'] = 0.0
            features['duration_max'] = 0.0
            features['duration_median'] = 0.0
        
        # Inter-node delays
        if 'inter_node_delays' in self.temporal_features and len(start_times) > 1:
            delays = []
            for u, v, edge_data in graph.edges(data=True):
                latency = edge_data.get('latency', 0.0)
                if latency > 0:
                    delays.append(latency)
            
            if delays:
                features['inter_node_delays_mean'] = np.mean(delays)
                features['inter_node_delays_std'] = np.std(delays)
            else:
                features['inter_node_delays_mean'] = 0.0
                features['inter_node_delays_std'] = 0.0
        
        # Execution gaps (large delays between nodes)
        if 'execution_gaps' in self.temporal_features:
            features['execution_gaps'] = self._count_execution_gaps(graph)
        
        # Concurrency levels
        if 'concurrency_levels' in self.temporal_features:
            features['concurrency_levels'] = self._estimate_concurrency(graph)
        
        # Operation order consistency
        if 'operation_order_consistency' in self.temporal_features:
            features['operation_order_consistency'] = self._check_temporal_consistency(graph)
        
        # Temporal anomalies
        if 'temporal_anomalies' in self.temporal_features:
            features['temporal_anomalies'] = self._detect_temporal_anomalies(graph)
        
        return features
    
    def _extract_semantic_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Extract semantic features related to agent behavior."""
        features = {}
        
        if len(graph.nodes) == 0:
            return {key: 0.0 for key in self.semantic_features}
        
        # Agent type distribution
        if 'agent_type_counts' in self.semantic_features:
            agent_counts = {}
            for node_id, node_data in graph.nodes(data=True):
                agent_type = node_data.get('agent_type', 'unknown')
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            
            # Convert to features
            for agent_type, count in agent_counts.items():
                features[f'agent_type_{agent_type}_count'] = count
                features[f'agent_type_{agent_type}_ratio'] = count / len(graph.nodes)
        
        # Handoff frequency
        if 'handoff_frequency' in self.semantic_features:
            handoff_count = sum(1 for _, node_data in graph.nodes(data=True) 
                              if node_data.get('node_type') == 'handoff')
            features['handoff_count'] = handoff_count
            features['handoff_frequency'] = handoff_count / len(graph.nodes)
        
        # Tool type distribution
        if 'tool_type_distribution' in self.semantic_features:
            tool_counts = {}
            tool_nodes = 0
            
            for node_id, node_data in graph.nodes(data=True):
                if node_data.get('node_type') == 'tool_call':
                    tool_nodes += 1
                    tool_type = node_data.get('tool_type', 'unknown')
                    tool_counts[tool_type] = tool_counts.get(tool_type, 0) + 1
            
            features['total_tool_calls'] = tool_nodes
            
            # Convert to features
            for tool_type, count in tool_counts.items():
                features[f'tool_type_{tool_type}_count'] = count
                if tool_nodes > 0:
                    features[f'tool_type_{tool_type}_ratio'] = count / tool_nodes
        
        # Tool failure rates
        if 'tool_failure_rates' in self.semantic_features:
            tool_failures = sum(1 for _, node_data in graph.nodes(data=True)
                              if (node_data.get('node_type') == 'tool_call' and 
                                  not node_data.get('tool_success', True)))
            
            tool_calls = sum(1 for _, node_data in graph.nodes(data=True)
                           if node_data.get('node_type') == 'tool_call')
            
            features['tool_failure_count'] = tool_failures
            features['tool_failure_rate'] = tool_failures / max(tool_calls, 1)
        
        # Error frequency
        if 'error_frequency' in self.semantic_features:
            error_count = sum(1 for _, node_data in graph.nodes(data=True)
                            if not node_data.get('success', True))
            
            features['error_count'] = error_count
            features['error_frequency'] = error_count / len(graph.nodes)
        
        # Error clustering
        if 'error_clustering' in self.semantic_features:
            features['error_clustering'] = self._calculate_error_clustering(graph)
        
        # Recovery patterns
        if 'recovery_patterns' in self.semantic_features:
            features.update(self._analyze_recovery_patterns(graph))
        
        # Node type distribution
        node_type_counts = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        for node_type, count in node_type_counts.items():
            features[f'node_type_{node_type}_count'] = count
            features[f'node_type_{node_type}_ratio'] = count / len(graph.nodes)
        
        # Edge type distribution
        edge_type_counts = {}
        for u, v, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        total_edges = len(graph.edges)
        for edge_type, count in edge_type_counts.items():
            features[f'edge_type_{edge_type}_count'] = count
            if total_edges > 0:
                features[f'edge_type_{edge_type}_ratio'] = count / total_edges
        
        return features
    
    def _calculate_centrality(self, graph: nx.DiGraph, centrality_type: str) -> Dict:
        """Calculate centrality measures for the graph."""
        try:
            if centrality_type == 'betweenness_centrality':
                return nx.betweenness_centrality(graph)
            elif centrality_type == 'closeness_centrality':
                return nx.closeness_centrality(graph)
            elif centrality_type == 'eigenvector_centrality':
                return nx.eigenvector_centrality(graph, max_iter=1000)
            else:
                return {}
        except:
            return {}
    
    def _calculate_longest_path(self, graph: nx.DiGraph) -> int:
        """Calculate the longest path in the graph."""
        try:
            if nx.is_directed_acyclic_graph(graph):
                return self._dag_longest_path(graph)
            else:
                # For cyclic graphs, use a different approach
                longest = 0
                for node in graph.nodes():
                    try:
                        paths = nx.single_source_shortest_path(graph, node, cutoff=50)
                        if paths:
                            max_length = max(len(path) - 1 for path in paths.values())
                            longest = max(longest, max_length)
                    except:
                        continue
                return longest
        except:
            return 0
    
    def _dag_longest_path(self, graph: nx.DiGraph) -> int:
        """Calculate longest path in a DAG."""
        try:
            return nx.dag_longest_path_length(graph)
        except:
            return 0
    
    def _calculate_dag_depth(self, graph: nx.DiGraph) -> int:
        """Calculate the depth (number of levels) in the DAG."""
        if not nx.is_directed_acyclic_graph(graph):
            return 0
        
        try:
            # Use topological generations
            generations = list(nx.topological_generations(graph))
            return len(generations)
        except:
            return 0
    
    def _calculate_level_statistics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate statistics about nodes per level."""
        features = {}
        
        try:
            if nx.is_directed_acyclic_graph(graph):
                generations = list(nx.topological_generations(graph))
                level_widths = [len(generation) for generation in generations]
                
                if level_widths:
                    features['nodes_per_level_mean'] = np.mean(level_widths)
                    features['nodes_per_level_std'] = np.std(level_widths)
                    features['level_width_variance'] = np.var(level_widths)
                else:
                    features['nodes_per_level_mean'] = 0.0
                    features['nodes_per_level_std'] = 0.0
                    features['level_width_variance'] = 0.0
            else:
                features['nodes_per_level_mean'] = 0.0
                features['nodes_per_level_std'] = 0.0
                features['level_width_variance'] = 0.0
        except:
            features['nodes_per_level_mean'] = 0.0
            features['nodes_per_level_std'] = 0.0
            features['level_width_variance'] = 0.0
        
        return features
    
    def _calculate_branching_factor(self, graph: nx.DiGraph) -> float:
        """Calculate average branching factor."""
        out_degrees = [graph.out_degree(node) for node in graph.nodes()]
        return np.mean(out_degrees) if out_degrees else 0.0
    
    def _count_parallel_paths(self, graph: nx.DiGraph) -> int:
        """Count the number of parallel paths in the graph."""
        # Simple heuristic: count nodes with multiple outgoing edges
        parallel_nodes = sum(1 for node in graph.nodes() if graph.out_degree(node) > 1)
        return parallel_nodes
    
    def _count_execution_gaps(self, graph: nx.DiGraph) -> int:
        """Count execution gaps (large delays between nodes)."""
        gap_count = 0
        gap_threshold = 10.0  # seconds
        
        for u, v, edge_data in graph.edges(data=True):
            latency = edge_data.get('latency', 0.0)
            if latency > gap_threshold:
                gap_count += 1
        
        return gap_count
    
    def _estimate_concurrency(self, graph: nx.DiGraph) -> float:
        """Estimate concurrency level in the graph."""
        # Simple heuristic: ratio of nodes with multiple predecessors
        concurrent_nodes = sum(1 for node in graph.nodes() if graph.in_degree(node) > 1)
        return concurrent_nodes / len(graph.nodes()) if len(graph.nodes()) > 0 else 0.0
    
    def _check_temporal_consistency(self, graph: nx.DiGraph) -> float:
        """Check if temporal ordering is consistent with graph structure."""
        violations = 0
        total_edges = 0
        
        for u, v, edge_data in graph.edges(data=True):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            u_time = u_data.get('start_timestamp', 0)
            v_time = v_data.get('start_timestamp', 0)
            
            if u_time > 0 and v_time > 0:
                total_edges += 1
                if u_time >= v_time:  # Temporal violation
                    violations += 1
        
        return 1.0 - (violations / max(total_edges, 1))
    
    def _detect_temporal_anomalies(self, graph: nx.DiGraph) -> int:
        """Detect temporal anomalies in the graph."""
        anomaly_count = 0
        
        # Check for nodes with anomalous durations
        durations = [node_data.get('duration', 0.0) for _, node_data in graph.nodes(data=True)]
        
        if len(durations) > 3:
            # Use z-score to detect outliers
            z_scores = np.abs(stats.zscore(durations))
            anomaly_count += np.sum(z_scores > 3)  # More than 3 standard deviations
        
        return anomaly_count
    
    def _calculate_error_clustering(self, graph: nx.DiGraph) -> float:
        """Calculate how clustered errors are in the graph."""
        error_nodes = [node for node, data in graph.nodes(data=True) 
                      if not data.get('success', True)]
        
        if len(error_nodes) < 2:
            return 0.0
        
        # Calculate average distance between error nodes
        total_distance = 0
        pairs = 0
        
        for i, node1 in enumerate(error_nodes):
            for node2 in error_nodes[i+1:]:
                try:
                    distance = nx.shortest_path_length(graph.to_undirected(), node1, node2)
                    total_distance += distance
                    pairs += 1
                except nx.NetworkXNoPath:
                    continue
        
        if pairs > 0:
            avg_distance = total_distance / pairs
            # Normalize by graph diameter
            try:
                diameter = nx.diameter(graph.to_undirected())
                return 1.0 - (avg_distance / max(diameter, 1))
            except:
                return 0.0
        
        return 0.0
    
    def _analyze_recovery_patterns(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Analyze error recovery patterns."""
        features = {}
        
        # Count nodes with retry attempts
        retry_nodes = sum(1 for _, node_data in graph.nodes(data=True)
                         if node_data.get('retry_count', 0) > 0)
        
        features['retry_attempts'] = retry_nodes
        features['retry_frequency'] = retry_nodes / len(graph.nodes())
        
        # Count successful recoveries
        recovery_nodes = 0
        for node_id, node_data in graph.nodes(data=True):
            if (node_data.get('retry_count', 0) > 0 and 
                node_data.get('success', True)):
                recovery_nodes += 1
        
        features['successful_recoveries'] = recovery_nodes
        features['recovery_success_rate'] = recovery_nodes / max(retry_nodes, 1)
        
        return features
    
    def _extract_graph_structure_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Extract raw graph structure features essential for GNNs.
        
        This method extracts:
        1. Adjacency matrix features (density, sparsity, connectivity patterns)
        2. Edge attribute statistics (latency, reliability, data transfer patterns)
        3. Graph-level structural properties (connectivity, centrality distributions)
        4. Node degree distributions and patterns
        5. Edge weight distributions and patterns
        """
        features = {}
        
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        
        if num_nodes == 0:
            return self._get_empty_structure_features()
        
        # 1. Adjacency Matrix Features
        adj_matrix = nx.adjacency_matrix(graph).toarray()
        features.update(self._extract_adjacency_features(adj_matrix))
        
        # 2. Edge Attribute Statistics
        features.update(self._extract_edge_attribute_features(graph))
        
        # 3. Node Degree Features
        features.update(self._extract_degree_features(graph))
        
        # 4. Graph Connectivity Features
        features.update(self._extract_connectivity_features(graph))
        
        # 5. Centrality Distribution Features
        features.update(self._extract_centrality_distribution_features(graph))
        
        # 6. Path and Distance Features
        features.update(self._extract_path_features(graph))
        
        # 7. Graph Topology Features
        features.update(self._extract_topology_features(graph))
        
        return features
    
    def _extract_adjacency_features(self, adj_matrix: np.ndarray) -> Dict[str, float]:
        """Extract features from adjacency matrix."""
        features = {}
        
        # Basic adjacency statistics
        features['adjacency_density'] = np.mean(adj_matrix)
        features['adjacency_sparsity'] = 1.0 - features['adjacency_density']
        features['adjacency_std'] = np.std(adj_matrix)
        features['adjacency_max'] = np.max(adj_matrix)
        features['adjacency_min'] = np.min(adj_matrix)
        
        # Connectivity patterns
        features['self_loops'] = np.sum(np.diag(adj_matrix))
        features['bidirectional_edges'] = np.sum((adj_matrix * adj_matrix.T) > 0) // 2
        
        # Row/column statistics
        features['row_means'] = np.mean(np.mean(adj_matrix, axis=1))
        features['row_stds'] = np.mean(np.std(adj_matrix, axis=1))
        features['col_means'] = np.mean(np.mean(adj_matrix, axis=0))
        features['col_stds'] = np.mean(np.std(adj_matrix, axis=0))
        
        return features
    
    def _extract_edge_attribute_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract comprehensive edge attribute statistics."""
        features = {}
        
        if len(graph.edges) == 0:
            return self._get_empty_edge_features()
        
        # Extract all edge attributes
        edge_attrs = {}
        for u, v, data in graph.edges(data=True):
            for key, value in data.items():
                if key not in edge_attrs:
                    edge_attrs[key] = []
                edge_attrs[key].append(value)
        
        # Calculate statistics for each edge attribute
        for attr_name, values in edge_attrs.items():
            if isinstance(values[0], (int, float)):
                values = [float(v) for v in values if v is not None]
                if values:
                    features[f'edge_{attr_name}_mean'] = np.mean(values)
                    features[f'edge_{attr_name}_std'] = np.std(values)
                    features[f'edge_{attr_name}_min'] = np.min(values)
                    features[f'edge_{attr_name}_max'] = np.max(values)
                    features[f'edge_{attr_name}_median'] = np.median(values)
                    features[f'edge_{attr_name}_q25'] = np.percentile(values, 25)
                    features[f'edge_{attr_name}_q75'] = np.percentile(values, 75)
        
        # Special handling for latency (critical for performance analysis)
        if 'latency' in edge_attrs:
            latencies = [float(v) for v in edge_attrs['latency'] if v is not None]
            if latencies:
                features['latency_mean'] = np.mean(latencies)
                features['latency_std'] = np.std(latencies)
                features['latency_max'] = np.max(latencies)
                features['latency_min'] = np.min(latencies)
                features['latency_median'] = np.median(latencies)
                features['latency_p95'] = np.percentile(latencies, 95)
                features['latency_p99'] = np.percentile(latencies, 99)
        
        # Edge type distribution
        if 'edge_type' in edge_attrs:
            edge_types = edge_attrs['edge_type']
            unique_types = set(edge_types)
            features['edge_type_diversity'] = len(unique_types)
            for edge_type in unique_types:
                count = edge_types.count(edge_type)
                features[f'edge_type_{edge_type}_count'] = count
                features[f'edge_type_{edge_type}_ratio'] = count / len(edge_types)
        
        # Reliability and error patterns
        if 'reliability_score' in edge_attrs:
            reliability_scores = [float(v) for v in edge_attrs['reliability_score'] if v is not None]
            if reliability_scores:
                features['reliability_mean'] = np.mean(reliability_scores)
                features['reliability_std'] = np.std(reliability_scores)
                features['reliability_min'] = np.min(reliability_scores)
        
        if 'error_count' in edge_attrs:
            error_counts = [int(v) for v in edge_attrs['error_count'] if v is not None]
            if error_counts:
                features['total_errors'] = sum(error_counts)
                features['error_rate'] = sum(error_counts) / len(graph.edges)
                features['error_std'] = np.std(error_counts)
        
        return features
    
    def _extract_degree_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract node degree distribution features."""
        features = {}
        
        # In-degree and out-degree
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        total_degrees = [d for n, d in graph.degree()]
        
        # In-degree statistics
        features['in_degree_mean'] = np.mean(in_degrees)
        features['in_degree_std'] = np.std(in_degrees)
        features['in_degree_max'] = np.max(in_degrees)
        features['in_degree_min'] = np.min(in_degrees)
        features['in_degree_median'] = np.median(in_degrees)
        
        # Out-degree statistics
        features['out_degree_mean'] = np.mean(out_degrees)
        features['out_degree_std'] = np.std(out_degrees)
        features['out_degree_max'] = np.max(out_degrees)
        features['out_degree_min'] = np.min(out_degrees)
        features['out_degree_median'] = np.median(out_degrees)
        
        # Total degree statistics
        features['total_degree_mean'] = np.mean(total_degrees)
        features['total_degree_std'] = np.std(total_degrees)
        features['total_degree_max'] = np.max(total_degrees)
        features['total_degree_min'] = np.min(total_degrees)
        
        # Degree distribution features
        features['degree_skewness'] = stats.skew(total_degrees) if len(total_degrees) > 2 else 0.0
        features['degree_kurtosis'] = stats.kurtosis(total_degrees) if len(total_degrees) > 3 else 0.0
        
        # Degree correlation
        features['degree_correlation'] = np.corrcoef(in_degrees, out_degrees)[0, 1] if len(in_degrees) > 1 else 0.0
        
        # Degree-based node counts
        features['leaf_nodes'] = sum(1 for d in in_degrees if d == 0)
        features['root_nodes'] = sum(1 for d in out_degrees if d == 0)
        features['isolated_nodes'] = sum(1 for d in total_degrees if d == 0)
        
        return features
    
    def _extract_connectivity_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract graph connectivity features."""
        features = {}
        
        # Connected components
        weakly_connected = list(nx.weakly_connected_components(graph))
        strongly_connected = list(nx.strongly_connected_components(graph))
        
        features['weakly_connected_components'] = len(weakly_connected)
        features['strongly_connected_components'] = len(strongly_connected)
        features['largest_weakly_connected_size'] = max(len(comp) for comp in weakly_connected) if weakly_connected else 0
        features['largest_strongly_connected_size'] = max(len(comp) for comp in strongly_connected) if strongly_connected else 0
        
        # Connectivity ratios
        features['weakly_connected_ratio'] = features['largest_weakly_connected_size'] / len(graph.nodes) if len(graph.nodes) > 0 else 0
        features['strongly_connected_ratio'] = features['largest_strongly_connected_size'] / len(graph.nodes) if len(graph.nodes) > 0 else 0
        
        # Node connectivity
        try:
            features['node_connectivity'] = nx.node_connectivity(graph)
        except:
            features['node_connectivity'] = 0
        
        # Edge connectivity
        try:
            features['edge_connectivity'] = nx.edge_connectivity(graph)
        except:
            features['edge_connectivity'] = 0
        
        return features
    
    def _extract_centrality_distribution_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract centrality distribution features."""
        features = {}
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(graph)
            betweenness_values = list(betweenness.values())
            features['betweenness_mean'] = np.mean(betweenness_values)
            features['betweenness_std'] = np.std(betweenness_values)
            features['betweenness_max'] = np.max(betweenness_values)
            features['betweenness_skewness'] = stats.skew(betweenness_values) if len(betweenness_values) > 2 else 0.0
        except:
            features['betweenness_mean'] = 0.0
            features['betweenness_std'] = 0.0
            features['betweenness_max'] = 0.0
            features['betweenness_skewness'] = 0.0
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(graph)
            closeness_values = list(closeness.values())
            features['closeness_mean'] = np.mean(closeness_values)
            features['closeness_std'] = np.std(closeness_values)
            features['closeness_max'] = np.max(closeness_values)
            features['closeness_skewness'] = stats.skew(closeness_values) if len(closeness_values) > 2 else 0.0
        except:
            features['closeness_mean'] = 0.0
            features['closeness_std'] = 0.0
            features['closeness_max'] = 0.0
            features['closeness_skewness'] = 0.0
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            eigenvector_values = list(eigenvector.values())
            features['eigenvector_mean'] = np.mean(eigenvector_values)
            features['eigenvector_std'] = np.std(eigenvector_values)
            features['eigenvector_max'] = np.max(eigenvector_values)
            features['eigenvector_skewness'] = stats.skew(eigenvector_values) if len(eigenvector_values) > 2 else 0.0
        except:
            features['eigenvector_mean'] = 0.0
            features['eigenvector_std'] = 0.0
            features['eigenvector_max'] = 0.0
            features['eigenvector_skewness'] = 0.0
        
        return features
    
    def _extract_path_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract path and distance features."""
        features = {}
        
        # Shortest path statistics
        try:
            if nx.is_weakly_connected(graph):
                undirected = graph.to_undirected()
                path_lengths = []
                for source in graph.nodes():
                    for target in graph.nodes():
                        if source != target:
                            try:
                                length = nx.shortest_path_length(undirected, source, target)
                                path_lengths.append(length)
                            except:
                                continue
                
                if path_lengths:
                    features['shortest_path_mean'] = np.mean(path_lengths)
                    features['shortest_path_std'] = np.std(path_lengths)
                    features['shortest_path_max'] = np.max(path_lengths)
                    features['shortest_path_min'] = np.min(path_lengths)
                    features['shortest_path_median'] = np.median(path_lengths)
                else:
                    features['shortest_path_mean'] = 0.0
                    features['shortest_path_std'] = 0.0
                    features['shortest_path_max'] = 0.0
                    features['shortest_path_min'] = 0.0
                    features['shortest_path_median'] = 0.0
            else:
                features['shortest_path_mean'] = np.inf
                features['shortest_path_std'] = 0.0
                features['shortest_path_max'] = np.inf
                features['shortest_path_min'] = np.inf
                features['shortest_path_median'] = np.inf
        except:
            features['shortest_path_mean'] = 0.0
            features['shortest_path_std'] = 0.0
            features['shortest_path_max'] = 0.0
            features['shortest_path_min'] = 0.0
            features['shortest_path_median'] = 0.0
        
        # Longest path in DAG
        features['longest_path_length'] = self._calculate_longest_path(graph)
        
        return features
    
    def _extract_topology_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Extract graph topology features."""
        features = {}
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(graph.to_undirected())
            clustering_values = list(clustering.values())
            features['clustering_mean'] = np.mean(clustering_values)
            features['clustering_std'] = np.std(clustering_values)
            features['clustering_max'] = np.max(clustering_values)
        except:
            features['clustering_mean'] = 0.0
            features['clustering_std'] = 0.0
            features['clustering_max'] = 0.0
        
        # Transitivity
        try:
            features['transitivity'] = nx.transitivity(graph.to_undirected())
        except:
            features['transitivity'] = 0.0
        
        # Assortativity
        try:
            features['degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
        except:
            features['degree_assortativity'] = 0.0
        
        # Graph diameter and radius
        try:
            if nx.is_weakly_connected(graph):
                undirected = graph.to_undirected()
                features['diameter'] = nx.diameter(undirected)
                features['radius'] = nx.radius(undirected)
                features['average_shortest_path_length'] = nx.average_shortest_path_length(undirected)
            else:
                features['diameter'] = np.inf
                features['radius'] = np.inf
                features['average_shortest_path_length'] = np.inf
        except:
            features['diameter'] = 0.0
            features['radius'] = 0.0
            features['average_shortest_path_length'] = 0.0
        
        return features
    
    def _get_empty_structure_features(self) -> Dict[str, float]:
        """Return empty structure features for graphs with no nodes."""
        return {
            'adjacency_density': 0.0, 'adjacency_sparsity': 1.0, 'adjacency_std': 0.0,
            'adjacency_max': 0.0, 'adjacency_min': 0.0, 'self_loops': 0.0,
            'bidirectional_edges': 0.0, 'row_means': 0.0, 'row_stds': 0.0,
            'col_means': 0.0, 'col_stds': 0.0, 'in_degree_mean': 0.0, 'in_degree_std': 0.0,
            'in_degree_max': 0.0, 'in_degree_min': 0.0, 'in_degree_median': 0.0,
            'out_degree_mean': 0.0, 'out_degree_std': 0.0, 'out_degree_max': 0.0,
            'out_degree_min': 0.0, 'out_degree_median': 0.0, 'total_degree_mean': 0.0,
            'total_degree_std': 0.0, 'total_degree_max': 0.0, 'total_degree_min': 0.0,
            'degree_skewness': 0.0, 'degree_kurtosis': 0.0, 'degree_correlation': 0.0,
            'leaf_nodes': 0.0, 'root_nodes': 0.0, 'isolated_nodes': 0.0,
            'weakly_connected_components': 0.0, 'strongly_connected_components': 0.0,
            'largest_weakly_connected_size': 0.0, 'largest_strongly_connected_size': 0.0,
            'weakly_connected_ratio': 0.0, 'strongly_connected_ratio': 0.0,
            'node_connectivity': 0.0, 'edge_connectivity': 0.0,
            'betweenness_mean': 0.0, 'betweenness_std': 0.0, 'betweenness_max': 0.0,
            'betweenness_skewness': 0.0, 'closeness_mean': 0.0, 'closeness_std': 0.0,
            'closeness_max': 0.0, 'closeness_skewness': 0.0, 'eigenvector_mean': 0.0,
            'eigenvector_std': 0.0, 'eigenvector_max': 0.0, 'eigenvector_skewness': 0.0,
            'shortest_path_mean': 0.0, 'shortest_path_std': 0.0, 'shortest_path_max': 0.0,
            'shortest_path_min': 0.0, 'shortest_path_median': 0.0, 'longest_path_length': 0.0,
            'clustering_mean': 0.0, 'clustering_std': 0.0, 'clustering_max': 0.0,
            'transitivity': 0.0, 'degree_assortativity': 0.0, 'diameter': 0.0,
            'radius': 0.0, 'average_shortest_path_length': 0.0
        }
    
    def _get_empty_edge_features(self) -> Dict[str, float]:
        """Return empty edge features for graphs with no edges."""
        return {
            'edge_latency_mean': 0.0, 'edge_latency_std': 0.0, 'edge_latency_min': 0.0,
            'edge_latency_max': 0.0, 'edge_latency_median': 0.0, 'edge_latency_q25': 0.0,
            'edge_latency_q75': 0.0, 'latency_mean': 0.0, 'latency_std': 0.0,
            'latency_max': 0.0, 'latency_min': 0.0, 'latency_median': 0.0,
            'latency_p95': 0.0, 'latency_p99': 0.0, 'edge_type_diversity': 0.0,
            'reliability_mean': 0.0, 'reliability_std': 0.0, 'reliability_min': 0.0,
            'total_errors': 0.0, 'error_rate': 0.0, 'error_std': 0.0
        }
