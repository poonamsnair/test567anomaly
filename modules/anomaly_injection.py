"""
Anomaly injection system for AI Agent Trajectory Anomaly Detection.

This module implements 10 specific anomaly types with configurable severity levels
to create realistic failure patterns in agent trajectories.
"""

import copy
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from .utils import (
    AgentTrajectory, AgentType, AnomalyType, NodeType, SeverityLevel,
    ToolType, TrajectoryEdge, TrajectoryNode, Timer
)

logger = logging.getLogger(__name__)


class AnomalyInjector:
    """
    Injects various types of anomalies into agent trajectories.
    
    This class implements 10 specific anomaly types:
    1. Infinite loops - Circular dependencies in execution flow
    2. Suboptimal paths - Unnecessarily complex routes
    3. Tool failure cascades - Multiple consecutive tool failures
    4. Planning paralysis - Excessive planning without execution
    5. Memory inconsistencies - Contradictory memory operations
    6. Timeout cascades - Chains of operations exceeding time limits
    7. Handoff failures - Failed context transfer between agents
    8. Validation loops - Excessive validation without resolution
    9. Context drift - Gradual deviation from original intent
    10. Incomplete responses - Trajectories ending without final output
    """
    
    def __init__(self, config: Dict, random_seed: int = 42):
        """
        Initialize anomaly injector.
        
        Args:
            config: Configuration dictionary
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Extract configuration
        self.anomaly_config = config.get('anomaly_injection', {})
        self.anomaly_types = self.anomaly_config.get('anomaly_types', {})
        self.severity_levels = self.anomaly_config.get('severity_levels', {})
        
        # Anomaly injection methods mapping
        self.injection_methods = {
            AnomalyType.INFINITE_LOOPS: self._inject_infinite_loops,
            AnomalyType.SUBOPTIMAL_PATHS: self._inject_suboptimal_paths,
            AnomalyType.TOOL_FAILURE_CASCADES: self._inject_tool_failure_cascades,
            AnomalyType.PLANNING_PARALYSIS: self._inject_planning_paralysis,
            AnomalyType.MEMORY_INCONSISTENCIES: self._inject_memory_inconsistencies,
            AnomalyType.TIMEOUT_CASCADES: self._inject_timeout_cascades,
            AnomalyType.HANDOFF_FAILURES: self._inject_handoff_failures,
            AnomalyType.VALIDATION_LOOPS: self._inject_validation_loops,
            AnomalyType.CONTEXT_DRIFT: self._inject_context_drift,
            AnomalyType.INCOMPLETE_RESPONSES: self._inject_incomplete_responses
        }
        
        logger.info("AnomalyInjector initialized with seed %d", random_seed)
    
    def inject_anomalies(self, normal_trajectories: List[AgentTrajectory], 
                        num_anomalous: int) -> List[AgentTrajectory]:
        """
        Inject anomalies into normal trajectories to create anomalous dataset.
        
        Args:
            normal_trajectories: List of normal trajectories
            num_anomalous: Number of anomalous trajectories to create
        
        Returns:
            List of anomalous trajectories
        """
        anomalous_trajectories = []
        
        logger.info("Injecting anomalies into %d trajectories", num_anomalous)
        
        with Timer() as timer:
            for i in tqdm(range(num_anomalous), desc="Injecting anomalies"):
                # Select a random normal trajectory to modify
                base_trajectory = random.choice(normal_trajectories)
                
                # Select anomaly type based on configured ratios
                anomaly_type = self._select_anomaly_type()
                
                # Create anomalous version
                anomalous_trajectory = self._create_anomalous_trajectory(
                    base_trajectory, anomaly_type
                )
                
                anomalous_trajectories.append(anomalous_trajectory)
        
        logger.info("Injected anomalies in %d trajectories in %.2f seconds", 
                   len(anomalous_trajectories), timer.elapsed())
        
        return anomalous_trajectories
    
    def _select_anomaly_type(self) -> AnomalyType:
        """Select anomaly type based on configured ratios."""
        anomaly_types = list(self.anomaly_types.keys())
        ratios = [self.anomaly_types[atype].get('ratio', 0.1) for atype in anomaly_types]
        
        # Normalize ratios
        total_ratio = sum(ratios)
        probabilities = [r / total_ratio for r in ratios]
        
        # Convert string keys to AnomalyType enum
        anomaly_type_str = np.random.choice(anomaly_types, p=probabilities)
        return AnomalyType(anomaly_type_str)
    
    def _create_anomalous_trajectory(self, base_trajectory: AgentTrajectory, 
                                   anomaly_type: AnomalyType) -> AgentTrajectory:
        """Create an anomalous version of a trajectory."""
        # Deep copy the base trajectory
        anomalous_trajectory = copy.deepcopy(base_trajectory)
        
        # Update trajectory metadata
        anomalous_trajectory.trajectory_id = str(uuid.uuid4())
        anomalous_trajectory.is_anomalous = True
        anomalous_trajectory.anomaly_types.add(anomaly_type)
        
        # Get severity level for this anomaly type
        severity = SeverityLevel(self.anomaly_types[anomaly_type.value]['severity'])
        anomalous_trajectory.anomaly_severity = severity
        
        # Apply the specific anomaly injection method
        injection_method = self.injection_methods[anomaly_type]
        injection_method(anomalous_trajectory, severity)
        
        # Post-process to ensure all nodes are connected
        self._ensure_connectivity(anomalous_trajectory)
        
        # Recalculate metrics
        anomalous_trajectory.calculate_metrics()
        
        return anomalous_trajectory
    
    def _ensure_connectivity(self, trajectory: AgentTrajectory) -> None:
        """Ensure all nodes in the trajectory are connected to prevent floating nodes."""
        if len(trajectory.nodes) < 2:
            return
        
        # Create a set of all node IDs
        all_node_ids = {node.node_id for node in trajectory.nodes}
        
        # Create a set of connected node IDs from edges
        connected_node_ids = set()
        for edge in trajectory.edges:
            connected_node_ids.add(edge.source_node_id)
            connected_node_ids.add(edge.target_node_id)
        
        # Find floating nodes (nodes without edges)
        floating_node_ids = all_node_ids - connected_node_ids
        
        if floating_node_ids:
            logger.warning(f"Found {len(floating_node_ids)} floating nodes, connecting them")
            
            # Connect floating nodes to the main graph
            for floating_id in floating_node_ids:
                # Find a random connected node to attach to
                if connected_node_ids:
                    target_id = random.choice(list(connected_node_ids))
                    
                    # Create edge from floating node to connected node
                    edge = TrajectoryEdge(
                        source_node_id=floating_id,
                        target_node_id=target_id,
                        edge_type="execution_flow",
                        relationship_type="floating_attach",
                        creation_time=datetime.now(),
                        is_anomalous=True,
                        anomaly_indicators=["floating_node_attachment"]
                    )
                    trajectory.edges.append(edge)
                    
                    # Add the floating node to connected set
                    connected_node_ids.add(floating_id)
                else:
                    # If no connected nodes, create a simple chain
                    if len(trajectory.nodes) >= 2:
                        # Find the floating node
                        floating_node = next(node for node in trajectory.nodes if node.node_id == floating_id)
                        # Find another node to connect to
                        other_node = next(node for node in trajectory.nodes if node.node_id != floating_id)
                        
                        edge = TrajectoryEdge(
                            source_node_id=floating_id,
                            target_node_id=other_node.node_id,
                            edge_type="execution_flow",
                            relationship_type="floating_attach",
                            creation_time=floating_node.start_time,
                            is_anomalous=True,
                            anomaly_indicators=["floating_node_attachment"]
                        )
                        trajectory.edges.append(edge)
                        connected_node_ids.add(floating_id)
    
    def _insert_anomalous_node_connected(self, trajectory, new_node):
        """Insert an anomalous node by splitting a random existing edge, or attach to a random node if no edges."""
        if trajectory.edges:
            # Pick a random edge to split
            edge_idx = random.randint(0, len(trajectory.edges) - 1)
            edge = trajectory.edges[edge_idx]
            # Remove the original edge
            trajectory.edges.pop(edge_idx)
            # Insert new_node between source and target
            trajectory.nodes.append(new_node)
            # Create new edges
            edge1 = type(edge)(
                source_node_id=edge.source_node_id,
                target_node_id=new_node.node_id,
                edge_type=edge.edge_type,
                relationship_type=edge.relationship_type,
                creation_time=edge.creation_time,
                is_anomalous=True,
                anomaly_indicators=getattr(edge, 'anomaly_indicators', [])
            )
            edge2 = type(edge)(
                source_node_id=new_node.node_id,
                target_node_id=edge.target_node_id,
                edge_type=edge.edge_type,
                relationship_type=edge.relationship_type,
                creation_time=edge.creation_time,
                is_anomalous=True,
                anomaly_indicators=getattr(edge, 'anomaly_indicators', [])
            )
            trajectory.edges.extend([edge1, edge2])
        else:
            # No edges: connect to a random node
            if trajectory.nodes:
                target = random.choice(trajectory.nodes)
                trajectory.nodes.append(new_node)
                edge = TrajectoryEdge(
                    source_node_id=new_node.node_id,
                    target_node_id=target.node_id,
                    edge_type="execution_flow",
                    relationship_type="anomaly_attach",
                    creation_time=new_node.start_time,
                    is_anomalous=True,
                    anomaly_indicators=["forced_attach"]
                )
                trajectory.edges.append(edge)
            else:
                trajectory.nodes.append(new_node)
    
    def _inject_infinite_loops(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject subtle loop anomaly - just a few repetitions, not infinite."""
        if len(trajectory.nodes) < 3:
            return
        
        # Determine loop characteristics based on severity - much more subtle
        if severity == SeverityLevel.CRITICAL:
            loop_size = min(len(trajectory.nodes) // 4, random.randint(2, 3))
            loop_count = random.randint(3, 5)
        elif severity == SeverityLevel.HIGH:
            loop_size = min(len(trajectory.nodes) // 5, random.randint(2, 3))
            loop_count = random.randint(2, 4)
        elif severity == SeverityLevel.MEDIUM:
            loop_size = min(len(trajectory.nodes) // 6, random.randint(1, 2))
            loop_count = random.randint(2, 3)
        else:  # LOW
            loop_size = min(len(trajectory.nodes) // 8, random.randint(1, 2))
            loop_count = random.randint(1, 2)
        
        # Select nodes to form the loop - choose a small section
        start_idx = random.randint(1, max(1, len(trajectory.nodes) - loop_size - 1))
        loop_nodes = trajectory.nodes[start_idx:start_idx + loop_size]
        
        # Create loop by repeating nodes
        current_time = loop_nodes[-1].end_time if loop_nodes and loop_nodes[-1].end_time else datetime.now()
        loop_trajectory_nodes = []
        
        for iteration in range(loop_count):
            for i, original_node in enumerate(loop_nodes):
                # Create duplicate node with updated timing
                loop_node = copy.deepcopy(original_node)
                loop_node.node_id = str(uuid.uuid4())
                loop_node.start_time = current_time
                loop_node.is_anomalous = True
                loop_node.anomaly_type = AnomalyType.INFINITE_LOOPS
                loop_node.anomaly_severity = severity
                loop_node.retry_count = iteration + 1
                
                # Add subtle loop metadata
                loop_node.anomaly_metadata = {
                    'loop_iteration': iteration + 1,
                    'loop_position': i,
                    'loop_size': loop_size,
                    'subtle_anomaly': True
                }
                
                # Slightly longer duration for loop nodes
                current_time += timedelta(seconds=random.uniform(1.0, 3.0))
                loop_node.end_time = current_time
                loop_node.calculate_duration()
                
                loop_trajectory_nodes.append(loop_node)
                
                # Connect to previous node in the main trajectory (linear progression)
                if i > 0:
                    # Connect to the previous loop node (creates linear chain, not cycle)
                    prev_loop_node = loop_trajectory_nodes[-2] if len(loop_trajectory_nodes) > 1 else None
                    if prev_loop_node:
                        edge = TrajectoryEdge(
                            source_node_id=prev_loop_node.node_id,
                            target_node_id=loop_node.node_id,
                            edge_type="execution_flow",
                            relationship_type="loop_sequence",
                            creation_time=current_time,
                            is_anomalous=True,
                            anomaly_indicators=["subtle_loop", "repetition"]
                        )
                        trajectory.edges.append(edge)
        
        # Insert loop nodes into trajectory
        trajectory.nodes.extend(loop_trajectory_nodes)
        
        # Update trajectory status - more subtle impact
        trajectory.completion_rate *= (1.0 - self.severity_levels[severity.value]['degradation_range'][0] * 0.3)
    
    def _inject_suboptimal_paths(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject subtle suboptimal path anomaly - just a few unnecessary steps, always connected."""
        if len(trajectory.nodes) < 3:
            return
        if severity == SeverityLevel.CRITICAL:
            extra_nodes = random.randint(2, 4)
        elif severity == SeverityLevel.HIGH:
            extra_nodes = random.randint(1, 3)
        elif severity == SeverityLevel.MEDIUM:
            extra_nodes = random.randint(1, 2)
        else:
            extra_nodes = random.randint(1, 1)
        current_time = datetime.now()
        for _ in range(extra_nodes):
            unnecessary_node = TrajectoryNode(
                node_type=NodeType.REASONING,
                agent_type=AgentType.PLANNER,
                start_time=current_time,
                description="Additional verification step",
                reasoning_type="double_check",
                reasoning_confidence=random.uniform(0.6, 0.8),
                is_anomalous=True,
                anomaly_type=AnomalyType.SUBOPTIMAL_PATHS,
                anomaly_severity=severity
            )
            unnecessary_node.anomaly_metadata = {
                'redundancy_reason': 'extra_verification',
                'optimal_skip': True,
                'subtle_anomaly': True
            }
            current_time += timedelta(seconds=random.uniform(0.5, 2.0))
            unnecessary_node.end_time = current_time
            unnecessary_node.calculate_duration()
            self._insert_anomalous_node_connected(trajectory, unnecessary_node)
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range']) * 0.2
        trajectory.completion_rate *= (1.0 - degradation)
    
    def _inject_tool_failure_cascades(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject subtle tool failure anomaly - occasional single failures, not cascades."""
        tool_nodes = [n for n in trajectory.nodes if n.node_type == NodeType.TOOL_CALL]
        if len(tool_nodes) < 1:
            return
        
        # Determine failure characteristics - much more subtle
        if severity == SeverityLevel.CRITICAL:
            failure_count = min(len(tool_nodes), random.randint(2, 3))
        elif severity == SeverityLevel.HIGH:
            failure_count = min(len(tool_nodes), random.randint(1, 2))
        elif severity == SeverityLevel.MEDIUM:
            failure_count = min(len(tool_nodes), random.randint(1, 1))
        else:  # LOW
            failure_count = min(len(tool_nodes), random.randint(1, 1))
        
        # Select random tool nodes to fail (not necessarily consecutive)
        failing_tools = random.sample(tool_nodes, failure_count)
        
        error_codes = ["TIMEOUT", "AUTH_FAILED", "RATE_LIMITED", "SERVICE_UNAVAILABLE", "INVALID_PARAMS"]
        
        for i, tool_node in enumerate(failing_tools):
            # Mark as failed but with subtle impact
            tool_node.tool_success = False
            tool_node.status = "failed"
            tool_node.is_anomalous = True
            tool_node.anomaly_type = AnomalyType.TOOL_FAILURE_CASCADES
            tool_node.anomaly_severity = severity
            tool_node.error_code = random.choice(error_codes)
            tool_node.error_message = f"Tool failed: {tool_node.error_code}"
            
            # Add subtle failure metadata
            tool_node.anomaly_metadata = {
                'failure_type': 'single_tool_failure',
                'failure_position': i,
                'total_failures': failure_count,
                'subtle_anomaly': True
            }
            
            # Add retry logic for subtle failures
            if random.random() < 0.7:  # 70% chance of retry
                tool_node.retry_count = 1
                
                # Update description to show retry
                tool_node.description = f"{tool_node.description} (retry successful)"
        
        # Update trajectory metrics - minimal impact
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range']) * 0.3
        trajectory.completion_rate *= (1.0 - degradation)
        trajectory.total_errors += failure_count
    
    def _inject_planning_paralysis(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject subtle planning inefficiency - slight over-planning, always connected."""
        planning_nodes = [n for n in trajectory.nodes if n.node_type == NodeType.PLANNING]
        if severity == SeverityLevel.CRITICAL:
            extra_planning = random.randint(2, 3)
        elif severity == SeverityLevel.HIGH:
            extra_planning = random.randint(1, 2)
        elif severity == SeverityLevel.MEDIUM:
            extra_planning = random.randint(1, 1)
        else:
            extra_planning = random.randint(1, 1)
        current_time = datetime.now()
        for i in range(extra_planning):
            planning_types = ["additional_analysis", "risk_assessment", "resource_check"]
            planning_node = TrajectoryNode(
                node_type=NodeType.PLANNING,
                agent_type=AgentType.PLANNER,
                start_time=current_time,
                description=f"Additional planning: {random.choice(planning_types)}",
                planning_type=random.choice(planning_types),
                planning_confidence=random.uniform(0.6, 0.8),
                is_anomalous=True,
                anomaly_type=AnomalyType.PLANNING_PARALYSIS,
                anomaly_severity=severity
            )
            planning_node.anomaly_metadata = {
                'extra_planning_step': i + 1,
                'total_extra_steps': extra_planning,
                'subtle_inefficiency': True
            }
            current_time += timedelta(seconds=random.uniform(2.0, 8.0))
            planning_node.end_time = current_time
            planning_node.calculate_duration()
            self._insert_anomalous_node_connected(trajectory, planning_node)
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range']) * 0.2
        trajectory.completion_rate *= (1.0 - degradation)
    
    def _inject_memory_inconsistencies(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject memory inconsistency anomaly."""
        memory_nodes = [n for n in trajectory.nodes if n.node_type == NodeType.MEMORY_ACCESS]
        
        # Add memory nodes if there aren't enough
        if len(memory_nodes) < 3:
            for i in range(5):
                memory_node = TrajectoryNode(
                    node_type=NodeType.MEMORY_ACCESS,
                    agent_type=AgentType.EXECUTOR,
                    start_time=datetime.now(),
                    description="Memory operation",
                    memory_operation=random.choice(["store", "retrieve", "update", "delete"]),
                    memory_key=f"key_{random.randint(1, 100)}",
                    memory_success=True
                )
                # Use connected insertion to ensure proper connectivity
                self._insert_anomalous_node_connected(trajectory, memory_node)
                memory_nodes.append(memory_node)
        
        # Determine inconsistency characteristics
        if severity == SeverityLevel.CRITICAL:
            inconsistency_count = random.randint(5, 10)
        elif severity == SeverityLevel.HIGH:
            inconsistency_count = random.randint(3, 6)
        elif severity == SeverityLevel.MEDIUM:
            inconsistency_count = random.randint(2, 4)
        else:  # LOW
            inconsistency_count = random.randint(1, 3)
        
        inconsistency_patterns = [
            "store_retrieve_mismatch",
            "double_delete",
            "update_nonexistent",
            "conflicting_updates",
            "circular_dependencies"
        ]
        
        for i in range(min(inconsistency_count, len(memory_nodes))):
            memory_node = random.choice(memory_nodes)
            pattern = random.choice(inconsistency_patterns)
            
            # Apply inconsistency pattern
            memory_node.is_anomalous = True
            memory_node.anomaly_type = AnomalyType.MEMORY_INCONSISTENCIES
            memory_node.anomaly_severity = severity
            memory_node.memory_success = False
            memory_node.status = "failed"
            memory_node.error_message = f"Memory inconsistency: {pattern}"
            
            memory_node.anomaly_metadata = {
                'inconsistency_pattern': pattern,
                'memory_corruption': True,
                'data_integrity_violation': True
            }
        
        # Update trajectory metrics
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range'])
        trajectory.completion_rate *= (1.0 - degradation)
        trajectory.total_errors += inconsistency_count
    
    def _inject_timeout_cascades(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject timeout cascade anomaly."""
        # Determine timeout characteristics
        if severity == SeverityLevel.CRITICAL:
            timeout_count = random.randint(8, 15)
            timeout_duration = random.uniform(30.0, 120.0)
        elif severity == SeverityLevel.HIGH:
            timeout_count = random.randint(5, 10)
            timeout_duration = random.uniform(20.0, 60.0)
        elif severity == SeverityLevel.MEDIUM:
            timeout_count = random.randint(3, 6)
            timeout_duration = random.uniform(10.0, 30.0)
        else:  # LOW
            timeout_count = random.randint(1, 4)
            timeout_duration = random.uniform(5.0, 15.0)
        
        # Select nodes to timeout
        candidate_nodes = [n for n in trajectory.nodes if n.node_type in [NodeType.TOOL_CALL, NodeType.REASONING]]
        timeout_nodes = random.sample(candidate_nodes, min(timeout_count, len(candidate_nodes)))
        
        for i, node in enumerate(timeout_nodes):
            node.status = "timeout"
            node.is_anomalous = True
            node.anomaly_type = AnomalyType.TIMEOUT_CASCADES
            node.anomaly_severity = severity
            node.error_code = "TIMEOUT"
            node.error_message = f"Operation timed out after {timeout_duration:.1f}s"
            
            # Significantly increase duration
            node.duration = timeout_duration
            if node.end_time and node.start_time:
                node.end_time = node.start_time + timedelta(seconds=timeout_duration)
            
            node.anomaly_metadata = {
                'timeout_position': i,
                'timeout_cascade_length': timeout_count,
                'timeout_duration': timeout_duration
            }
        
        # Update edges to reflect timeout propagation
        for edge in trajectory.edges:
            if any(edge.source_node_id == node.node_id for node in timeout_nodes):
                edge.timeout_count = random.randint(1, 3)
                edge.latency = timeout_duration
                edge.is_anomalous = True
                edge.anomaly_indicators.append("timeout_cascade")
        
        # Update trajectory metrics
        trajectory.total_errors += timeout_count
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range'])
        trajectory.completion_rate *= (1.0 - degradation)
        if severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            trajectory.success = False
    
    def _inject_handoff_failures(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject handoff failure anomaly."""
        handoff_nodes = [n for n in trajectory.nodes if n.node_type == NodeType.HANDOFF]
        
        # Add handoff nodes if there aren't enough - ensure they're connected
        if len(handoff_nodes) < 2:
            agents = list(AgentType)
            for i in range(4):
                source_agent = random.choice(agents)
                target_agent = random.choice([a for a in agents if a != source_agent])
                
                handoff_node = TrajectoryNode(
                    node_type=NodeType.HANDOFF,
                    agent_type=AgentType.COORDINATOR,
                    start_time=datetime.now(),
                    description=f"Handoff from {source_agent.value} to {target_agent.value}",
                    source_agent=source_agent,
                    target_agent=target_agent,
                    handoff_success=True
                )
                handoff_node.end_time = handoff_node.start_time + timedelta(seconds=random.uniform(1.0, 5.0))
                handoff_node.calculate_duration()
                
                # Use the connected insertion method to ensure proper connectivity
                self._insert_anomalous_node_connected(trajectory, handoff_node)
                handoff_nodes.append(handoff_node)
        
        # Determine failure characteristics
        if severity == SeverityLevel.CRITICAL:
            failure_count = len(handoff_nodes)  # All handoffs fail
        elif severity == SeverityLevel.HIGH:
            failure_count = max(1, len(handoff_nodes) * 3 // 4)
        elif severity == SeverityLevel.MEDIUM:
            failure_count = max(1, len(handoff_nodes) // 2)
        else:  # LOW
            failure_count = max(1, len(handoff_nodes) // 4)
        
        failing_handoffs = random.sample(handoff_nodes, failure_count)
        
        failure_reasons = [
            "context_loss",
            "agent_unavailable",
            "permission_denied",
            "serialization_error",
            "communication_timeout"
        ]
        
        for handoff_node in failing_handoffs:
            handoff_node.handoff_success = False
            handoff_node.status = "failed"
            handoff_node.is_anomalous = True
            handoff_node.anomaly_type = AnomalyType.HANDOFF_FAILURES
            handoff_node.anomaly_severity = severity
            
            failure_reason = random.choice(failure_reasons)
            handoff_node.error_code = f"HANDOFF_{failure_reason.upper()}"
            handoff_node.error_message = f"Handoff failed: {failure_reason}"
            
            handoff_node.anomaly_metadata = {
                'failure_reason': failure_reason,
                'context_preserved': False,
                'recovery_attempted': random.random() < 0.5
            }
            
            # Increase handoff duration due to failure
            if handoff_node.duration:
                handoff_node.duration *= random.uniform(3.0, 10.0)
        
        # Update trajectory metrics
        trajectory.total_handoffs = len(handoff_nodes)
        trajectory.total_errors += failure_count
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range'])
        trajectory.completion_rate *= (1.0 - degradation)
        
        if severity == SeverityLevel.CRITICAL:
            trajectory.success = False
    
    def _inject_validation_loops(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject validation loop anomaly."""
        validation_nodes = [n for n in trajectory.nodes if n.node_type == NodeType.VALIDATION]
        
        # Determine loop characteristics
        if severity == SeverityLevel.CRITICAL:
            loop_iterations = random.randint(10, 20)
        elif severity == SeverityLevel.HIGH:
            loop_iterations = random.randint(6, 12)
        elif severity == SeverityLevel.MEDIUM:
            loop_iterations = random.randint(3, 8)
        else:  # LOW
            loop_iterations = random.randint(2, 5)
        
        # Create validation loop
        if validation_nodes:
            base_validation = random.choice(validation_nodes)
        else:
            # Create a validation node if none exist - ensure it's connected
            base_validation = TrajectoryNode(
                node_type=NodeType.VALIDATION,
                agent_type=AgentType.VALIDATOR,
                start_time=datetime.now(),
                description="Quality validation",
                validation_result=False  # Always fails to create loop
            )
            base_validation.end_time = base_validation.start_time + timedelta(seconds=random.uniform(2.0, 8.0))
            base_validation.calculate_duration()
            
            # Use the connected insertion method to ensure proper connectivity
            self._insert_anomalous_node_connected(trajectory, base_validation)
        
        current_time = base_validation.end_time or datetime.now()
        
        # Create loop of failing validations
        for i in range(loop_iterations):
            validation_node = copy.deepcopy(base_validation)
            validation_node.node_id = str(uuid.uuid4())
            validation_node.start_time = current_time
            validation_node.validation_result = False  # Always fails
            validation_node.validation_score = random.uniform(0.1, 0.4)  # Low score
            validation_node.is_anomalous = True
            validation_node.anomaly_type = AnomalyType.VALIDATION_LOOPS
            validation_node.anomaly_severity = severity
            validation_node.retry_count = i + 1
            
            validation_node.anomaly_metadata = {
                'loop_iteration': i + 1,
                'total_iterations': loop_iterations,
                'validation_failure': True,
                'criteria_never_met': True
            }
            
            current_time += timedelta(seconds=random.uniform(2.0, 8.0))
            validation_node.end_time = current_time
            validation_node.calculate_duration()
            
            # Use the connected insertion method to ensure proper connectivity
            self._insert_anomalous_node_connected(trajectory, validation_node)
        
        # Update trajectory metrics
        degradation = random.uniform(*self.severity_levels[severity.value]['degradation_range'])
        trajectory.completion_rate *= (1.0 - degradation * 0.6)
    
    def _inject_context_drift(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject context drift anomaly."""
        # Determine drift characteristics
        if severity == SeverityLevel.CRITICAL:
            drift_intensity = random.uniform(0.8, 1.0)
        elif severity == SeverityLevel.HIGH:
            drift_intensity = random.uniform(0.6, 0.8)
        elif severity == SeverityLevel.MEDIUM:
            drift_intensity = random.uniform(0.4, 0.6)
        else:  # LOW
            drift_intensity = random.uniform(0.2, 0.4)
        
        # Apply gradual context drift
        original_task = trajectory.task_description
        drift_keywords = [
            "unrelated", "tangential", "off-topic", "divergent", 
            "alternative", "peripheral", "secondary", "incidental"
        ]
        
        # Gradually modify node descriptions and content
        drift_factor = 0.0
        for i, node in enumerate(trajectory.nodes):
            drift_factor = min(drift_intensity, (i / len(trajectory.nodes)) * drift_intensity)
            
            if random.random() < drift_factor:
                node.is_anomalous = True
                node.anomaly_type = AnomalyType.CONTEXT_DRIFT
                node.anomaly_severity = severity
                
                # Modify description to be less relevant
                drift_word = random.choice(drift_keywords)
                node.description = f"{drift_word} {node.description}"
                
                # Add drift metadata
                node.anomaly_metadata = {
                    'drift_factor': drift_factor,
                    'original_relevance': 1.0 - drift_factor,
                    'context_distance': drift_factor
                }
                
                # Modify tool parameters to be less relevant
                if node.node_type == NodeType.TOOL_CALL and node.tool_parameters:
                    for key in node.tool_parameters:
                        if isinstance(node.tool_parameters[key], str):
                            node.tool_parameters[key] = f"drift_{node.tool_parameters[key]}"
        
        # Update trajectory task description to reflect drift
        if drift_intensity > 0.5:
            drift_task = f"Drifted from original intent: {original_task}"
            trajectory.task_description = drift_task
        
        # Update trajectory metrics
        degradation = drift_intensity * 0.5  # Moderate impact
        trajectory.completion_rate *= (1.0 - degradation)
    
    def _inject_incomplete_responses(self, trajectory: AgentTrajectory, severity: SeverityLevel) -> None:
        """Inject incomplete response anomaly - ensure remaining nodes stay connected."""
        # Determine incompleteness characteristics
        if severity == SeverityLevel.CRITICAL:
            completion_rate = random.uniform(0.2, 0.4)  # Only 20-40% completed
        elif severity == SeverityLevel.HIGH:
            completion_rate = random.uniform(0.4, 0.6)  # 40-60% completed
        elif severity == SeverityLevel.MEDIUM:
            completion_rate = random.uniform(0.6, 0.8)  # 60-80% completed
        else:  # LOW
            completion_rate = random.uniform(0.8, 0.9)  # 80-90% completed
        
        # Calculate how many nodes to keep - ensure at least 2 nodes remain for connectivity
        nodes_to_keep = max(2, int(len(trajectory.nodes) * completion_rate))
        
        # Remove nodes from the end to simulate incomplete execution
        removed_nodes = trajectory.nodes[nodes_to_keep:]
        trajectory.nodes = trajectory.nodes[:nodes_to_keep]
        
        # Remove edges that reference removed nodes
        removed_node_ids = {node.node_id for node in removed_nodes}
        trajectory.edges = [
            edge for edge in trajectory.edges
            if edge.source_node_id not in removed_node_ids and 
               edge.target_node_id not in removed_node_ids
        ]
        
        # Ensure the remaining nodes are still connected by creating a simple linear chain
        # if there are gaps in connectivity
        if len(trajectory.nodes) >= 2:
            # Create a simple linear chain for remaining nodes
            new_edges = []
            for i in range(len(trajectory.nodes) - 1):
                edge = TrajectoryEdge(
                    source_node_id=trajectory.nodes[i].node_id,
                    target_node_id=trajectory.nodes[i + 1].node_id,
                    edge_type="execution_flow",
                    relationship_type="incomplete_chain",
                    creation_time=trajectory.nodes[i + 1].start_time,
                    is_anomalous=True,
                    anomaly_indicators=["incomplete_execution"]
                )
                new_edges.append(edge)
            
            # Replace all edges with the new linear chain
            trajectory.edges = new_edges
        
        # Mark the trajectory as incomplete
        trajectory.status = "incomplete"
        trajectory.success = False
        trajectory.completion_rate = completion_rate
        trajectory.is_anomalous = True
        trajectory.anomaly_types.add(AnomalyType.INCOMPLETE_RESPONSES)
        trajectory.anomaly_severity = severity
        
        # Mark remaining nodes as potentially anomalous
        if trajectory.nodes:
            last_node = trajectory.nodes[-1]
            last_node.is_anomalous = True
            last_node.anomaly_type = AnomalyType.INCOMPLETE_RESPONSES
            last_node.anomaly_severity = severity
            last_node.status = "interrupted"
            
            last_node.anomaly_metadata = {
                'incomplete_execution': True,
                'completion_percentage': completion_rate * 100,
                'remaining_steps': len(removed_nodes)
            }
        
        trajectory.anomaly_metadata = {
            'incompleteness_reason': 'premature_termination',
            'expected_nodes': len(trajectory.nodes) + len(removed_nodes),
            'actual_nodes': len(trajectory.nodes),
            'completion_percentage': completion_rate * 100
        }
    
    def get_anomaly_statistics(self, anomalous_trajectories: List[AgentTrajectory]) -> Dict:
        """Get statistics about injected anomalies."""
        stats = {
            'total_anomalous': len(anomalous_trajectories),
            'by_type': {},
            'by_severity': {},
            'avg_degradation': 0.0
        }
        
        # Count by type and severity
        for trajectory in anomalous_trajectories:
            for anomaly_type in trajectory.anomaly_types:
                stats['by_type'][anomaly_type.value] = stats['by_type'].get(anomaly_type.value, 0) + 1
            
            if trajectory.anomaly_severity:
                severity = trajectory.anomaly_severity.value
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        # Calculate average degradation
        degradations = []
        for trajectory in anomalous_trajectories:
            if trajectory.completion_rate < 1.0:
                degradations.append(1.0 - trajectory.completion_rate)
        
        if degradations:
            stats['avg_degradation'] = np.mean(degradations)
        
        return stats
