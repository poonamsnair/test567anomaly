"""
Utility functions and core data structures for AI Agent Trajectory Anomaly Detection System.

This module contains the fundamental data structures (TrajectoryNode, TrajectoryEdge) and
utility functions used throughout the system.
"""

import logging
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import yaml


class NodeType(Enum):
    """Types of nodes in agent trajectory graphs."""
    USER_ENQUIRY = "user_enquiry"  # User's initial request or question
    USER_INTERACTION = "user_interaction"  # Ongoing user interactions
    LLM_CALL = "llm_call"  # Agent using LLM for thinking/planning/response
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    MEMORY_ACCESS = "memory_access"
    PLANNING = "planning"
    VALIDATION = "validation"
    HANDOFF = "handoff"
    OBSERVATION = "observation"
    ACTION = "action"


class AgentType(Enum):
    """Types of agents in the system."""
    PLANNER = "Planner"
    EXECUTOR = "Executor"
    VALIDATOR = "Validator"
    COORDINATOR = "Coordinator"


class ToolType(Enum):
    """Types of tools available to agents."""
    WEB_SEARCH = "web_search"
    READ_DOCUMENT = "read_document"
    PREPARE_DOCUMENT = "prepare_document"
    ANALYZE_DATA = "analyze_data"
    DEEP_RESEARCH = "deep_research"
    WRITE_CODE = "write_code"
    MEMORY_STORE_RETRIEVE = "memory_store_retrieve"
    EXTERNAL_API = "external_api"


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    INFINITE_LOOPS = "infinite_loops"
    SUBOPTIMAL_PATHS = "suboptimal_paths"
    TOOL_FAILURE_CASCADES = "tool_failure_cascades"
    PLANNING_PARALYSIS = "planning_paralysis"
    MEMORY_INCONSISTENCIES = "memory_inconsistencies"
    TIMEOUT_CASCADES = "timeout_cascades"
    HANDOFF_FAILURES = "handoff_failures"
    VALIDATION_LOOPS = "validation_loops"
    CONTEXT_DRIFT = "context_drift"
    INCOMPLETE_RESPONSES = "incomplete_responses"


class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrajectoryNode:
    """
    Represents a single node in an agent trajectory graph.
    
    Each node represents a specific action or state in the agent's execution,
    such as tool calls, reasoning steps, memory access, planning, validation,
    handoffs between agents, observations, or final actions.
    """
    
    # Core identifiers
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.ACTION
    agent_type: AgentType = AgentType.EXECUTOR
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # Duration in seconds
    
    # Content and context
    description: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Tool-specific information (if node_type is TOOL_CALL)
    tool_type: Optional[ToolType] = None
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    tool_result: Dict[str, Any] = field(default_factory=dict)
    tool_success: bool = True
    tool_error_message: Optional[str] = None
    
    # Memory-specific information (if node_type is MEMORY_ACCESS)
    memory_operation: Optional[str] = None  # "store", "retrieve", "update", "delete"
    memory_key: Optional[str] = None
    memory_value: Any = None
    memory_success: bool = True
    
    # Planning-specific information (if node_type is PLANNING)
    planning_type: Optional[str] = None  # "task_decomposition", "strategy_selection", "resource_allocation"
    planning_output: Dict[str, Any] = field(default_factory=dict)
    planning_confidence: Optional[float] = None
    
    # Validation-specific information (if node_type is VALIDATION)
    validation_target: Optional[str] = None  # What is being validated
    validation_criteria: List[str] = field(default_factory=list)
    validation_result: bool = True
    validation_score: Optional[float] = None
    
    # Handoff-specific information (if node_type is HANDOFF)
    source_agent: Optional[AgentType] = None
    target_agent: Optional[AgentType] = None
    handoff_context: Dict[str, Any] = field(default_factory=dict)
    handoff_success: bool = True
    
    # Reasoning-specific information (if node_type is REASONING)
    reasoning_type: Optional[str] = None  # "analysis", "inference", "decision_making"
    reasoning_input: Dict[str, Any] = field(default_factory=dict)
    reasoning_output: Dict[str, Any] = field(default_factory=dict)
    reasoning_confidence: Optional[float] = None
    
    # Observation-specific information (if node_type is OBSERVATION)
    observation_source: Optional[str] = None
    observation_data: Dict[str, Any] = field(default_factory=dict)
    observation_timestamp: Optional[datetime] = None
    
    # User interaction-specific information (if node_type is USER_ENQUIRY or USER_INTERACTION)
    user_input: Optional[str] = None
    user_intent: Optional[str] = None  # "question", "clarification", "feedback", "approval"
    user_context: Dict[str, Any] = field(default_factory=dict)
    user_satisfaction: Optional[float] = None  # 0-1 scale
    
    # LLM call-specific information (if node_type is LLM_CALL)
    llm_model: Optional[str] = None  # "gpt-4", "claude-3", etc.
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    llm_purpose: Optional[str] = None  # "planning", "reasoning", "response_generation", "analysis"
    llm_confidence: Optional[float] = None
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    network_calls: int = 0
    
    # Status and error information
    status: str = "completed"  # "pending", "running", "completed", "failed", "timeout"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Anomaly injection metadata (for synthetic data)
    is_anomalous: bool = False
    anomaly_type: Optional[AnomalyType] = None
    anomaly_severity: Optional[SeverityLevel] = None
    anomaly_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Calculate duration if end_time is provided
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        # Set default observation timestamp
        if self.node_type == NodeType.OBSERVATION and not self.observation_timestamp:
            self.observation_timestamp = self.start_time
    
    def calculate_duration(self) -> float:
        """Calculate and return the duration of this node's execution."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
            return self.duration
        return 0.0
    
    def is_failed(self) -> bool:
        """Check if this node represents a failed operation."""
        return (self.status == "failed" or 
                not self.tool_success or 
                not self.memory_success or 
                not self.handoff_success or 
                not self.validation_result)
    
    def get_performance_score(self) -> float:
        """Calculate a performance score for this node (0-1, higher is better)."""
        base_score = 1.0
        
        # Reduce score for failures
        if self.is_failed():
            base_score *= 0.1
        
        # Reduce score for retries
        if self.retry_count > 0:
            base_score *= (1.0 - min(0.5, self.retry_count * 0.1))
        
        # Reduce score for timeouts
        if self.status == "timeout":
            base_score *= 0.2
        
        # Adjust for confidence scores
        if hasattr(self, 'planning_confidence') and self.planning_confidence:
            base_score *= self.planning_confidence
        elif hasattr(self, 'reasoning_confidence') and self.reasoning_confidence:
            base_score *= self.reasoning_confidence
        elif hasattr(self, 'validation_score') and self.validation_score:
            base_score *= self.validation_score
        
        return max(0.0, min(1.0, base_score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'agent_type': self.agent_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'description': self.description,
            'content': self.content,
            'tool_type': self.tool_type.value if self.tool_type else None,
            'tool_success': self.tool_success,
            'status': self.status,
            'is_anomalous': self.is_anomalous,
            'anomaly_type': self.anomaly_type.value if self.anomaly_type else None,
            'performance_score': self.get_performance_score()
        }


@dataclass
class TrajectoryEdge:
    """
    Represents an edge (connection) between two nodes in an agent trajectory graph.
    
    Edges represent the flow of execution, data dependencies, and relationships
    between different actions or states in the agent's trajectory.
    """
    
    # Core identifiers
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    
    # Edge type and semantics
    edge_type: str = "execution_flow"  # execution_flow, data_dependency, conditional, parallel, handoff
    relationship_type: str = "sequential"  # sequential, parallel, conditional, dependency
    
    # Timing and latency
    creation_time: datetime = field(default_factory=datetime.now)
    latency: Optional[float] = None  # Time delay between source and target
    
    # Data flow information
    data_transferred: Dict[str, Any] = field(default_factory=dict)
    data_size_bytes: Optional[int] = None
    data_type: Optional[str] = None
    
    # Conditional execution
    condition: Optional[str] = None
    condition_result: Optional[bool] = None
    
    # Probability and confidence
    probability: float = 1.0  # Probability of this edge being taken
    confidence: float = 1.0  # Confidence in the edge relationship
    
    # Performance metrics
    success_rate: float = 1.0
    error_count: int = 0
    timeout_count: int = 0
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Anomaly information
    is_anomalous: bool = False
    anomaly_indicators: List[str] = field(default_factory=list)
    
    # Priority and weight
    priority: int = 0
    weight: float = 1.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Validate edge consistency
        if not self.source_node_id or not self.target_node_id:
            raise ValueError("Source and target node IDs must be provided")
    
    def is_failed(self) -> bool:
        """Check if this edge represents a failed connection."""
        return (self.success_rate < 1.0 or 
                self.error_count > 0 or 
                self.timeout_count > 0)
    
    def get_reliability_score(self) -> float:
        """Calculate reliability score for this edge (0-1, higher is better)."""
        base_score = self.success_rate
        
        # Reduce score for errors and timeouts
        if self.error_count > 0:
            base_score *= (1.0 - min(0.5, self.error_count * 0.1))
        
        if self.timeout_count > 0:
            base_score *= (1.0 - min(0.3, self.timeout_count * 0.05))
        
        # Factor in confidence
        base_score *= self.confidence
        
        return max(0.0, min(1.0, base_score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            'edge_id': self.edge_id,
            'source_node_id': self.source_node_id,
            'target_node_id': self.target_node_id,
            'edge_type': self.edge_type,
            'relationship_type': self.relationship_type,
            'latency': self.latency,
            'probability': self.probability,
            'confidence': self.confidence,
            'success_rate': self.success_rate,
            'is_anomalous': self.is_anomalous,
            'reliability_score': self.get_reliability_score()
        }


@dataclass
class AgentTrajectory:
    """
    Represents a complete agent trajectory containing nodes and edges.
    
    A trajectory represents the complete execution path of an agent or
    set of agents working together to complete a task.
    """
    
    # Core identifiers
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    
    # Trajectory components
    nodes: List[TrajectoryNode] = field(default_factory=list)
    edges: List[TrajectoryEdge] = field(default_factory=list)
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    # Metadata
    agent_types_involved: Set[AgentType] = field(default_factory=set)
    tool_types_used: Set[ToolType] = field(default_factory=set)
    
    # Status and completion
    status: str = "completed"  # "pending", "running", "completed", "failed", "timeout"
    completion_rate: float = 1.0
    success: bool = True
    
    # Performance metrics
    total_tool_calls: int = 0
    total_handoffs: int = 0
    total_errors: int = 0
    average_node_duration: Optional[float] = None
    
    # Anomaly information
    is_anomalous: bool = False
    anomaly_types: Set[AnomalyType] = field(default_factory=set)
    anomaly_severity: Optional[SeverityLevel] = None
    anomaly_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    tags: Set[str] = field(default_factory=set)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: TrajectoryNode) -> None:
        """Add a node to the trajectory."""
        self.nodes.append(node)
        
        # Update metadata
        self.agent_types_involved.add(node.agent_type)
        if node.tool_type:
            self.tool_types_used.add(node.tool_type)
        
        # Update anomaly information
        if node.is_anomalous:
            self.is_anomalous = True
            if node.anomaly_type:
                self.anomaly_types.add(node.anomaly_type)
    
    def add_edge(self, edge: TrajectoryEdge) -> None:
        """Add an edge to the trajectory."""
        self.edges.append(edge)
        
        # Update anomaly information
        if edge.is_anomalous:
            self.is_anomalous = True
    
    def calculate_metrics(self) -> None:
        """Calculate and update trajectory metrics."""
        if not self.nodes:
            return
        
        # Calculate timing metrics
        if self.end_time and self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate node-based metrics
        durations = [node.duration for node in self.nodes if node.duration]
        if durations:
            self.average_node_duration = np.mean(durations)
        
        # Count different types of operations
        self.total_tool_calls = sum(1 for node in self.nodes if node.node_type == NodeType.TOOL_CALL)
        self.total_handoffs = sum(1 for node in self.nodes if node.node_type == NodeType.HANDOFF)
        self.total_errors = sum(1 for node in self.nodes if node.is_failed())
        
        # Calculate completion rate
        completed_nodes = sum(1 for node in self.nodes if node.status == "completed")
        self.completion_rate = completed_nodes / len(self.nodes) if self.nodes else 0.0
        
        # Determine overall success
        self.success = (self.completion_rate > 0.8 and 
                       self.status == "completed" and 
                       self.total_errors / len(self.nodes) < 0.2 if self.nodes else True)
    
    def get_node_by_id(self, node_id: str) -> Optional[TrajectoryNode]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_edges_from_node(self, node_id: str) -> List[TrajectoryEdge]:
        """Get all edges originating from a specific node."""
        return [edge for edge in self.edges if edge.source_node_id == node_id]
    
    def get_edges_to_node(self, node_id: str) -> List[TrajectoryEdge]:
        """Get all edges pointing to a specific node."""
        return [edge for edge in self.edges if edge.target_node_id == node_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary representation."""
        return {
            'trajectory_id': self.trajectory_id,
            'task_description': self.task_description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'status': self.status,
            'success': self.success,
            'completion_rate': self.completion_rate,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_tool_calls': self.total_tool_calls,
            'total_handoffs': self.total_handoffs,
            'total_errors': self.total_errors,
            'is_anomalous': self.is_anomalous,
            'anomaly_types': [at.value for at in self.anomaly_types],
            'anomaly_severity': self.anomaly_severity.value if self.anomaly_severity else None,
            'agent_types': [at.value for at in self.agent_types_involved],
            'tool_types': [tt.value for tt in self.tool_types_used]
        }


class Timer:
    """Simple timer utility for performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_graph_hash(nodes: List[TrajectoryNode], edges: List[TrajectoryEdge]) -> str:
    """
    Calculate a hash for a graph structure for comparison purposes.
    
    Args:
        nodes: List of trajectory nodes
        edges: List of trajectory edges
    
    Returns:
        Hash string representing the graph structure
    """
    import hashlib
    
    # Create a string representation of the graph structure
    node_string = "|".join(sorted([f"{n.node_id}:{n.node_type.value}:{n.agent_type.value}" for n in nodes]))
    edge_string = "|".join(sorted([f"{e.source_node_id}->{e.target_node_id}:{e.edge_type}" for e in edges]))
    
    graph_string = f"nodes:{node_string}|edges:{edge_string}"
    
    return hashlib.md5(graph_string.encode()).hexdigest()
