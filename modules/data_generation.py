"""
Synthetic data generation for AI Agent Trajectory Anomaly Detection System.

This module generates realistic agent trajectories with varying complexity patterns,
from simple linear executions to complex multi-agent workflows.
"""

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


class TrajectoryGenerator:
    """
    Generates synthetic agent trajectories for training and evaluation.
    
    This class creates realistic agent execution patterns including:
    - Simple linear task execution
    - Branched analysis workflows
    - Multi-agent handoffs and coordination
    - Complex research and analysis patterns
    - Error recovery scenarios
    """
    
    def __init__(self, config: Dict, random_seed: int = 42):
        """
        Initialize trajectory generator.
        
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
        self.data_config = config.get('data_generation', {})
        self.patterns = self.data_config.get('trajectory_patterns', {})
        
        # Task templates for realistic scenarios
        self.task_templates = self._initialize_task_templates()
        
        # Tool transition probabilities (realistic workflow patterns)
        self.tool_transitions = self._initialize_tool_transitions()
        
        logger.info("TrajectoryGenerator initialized with seed %d", random_seed)
    
    def _initialize_task_templates(self) -> Dict[str, List[str]]:
        """Initialize realistic task description templates."""
        return {
            'research': [
                "Analyze market trends for renewable energy sector",
                "Research competitor strategies in fintech industry",
                "Investigate best practices for remote work productivity",
                "Study impact of AI on healthcare outcomes",
                "Examine consumer behavior patterns in e-commerce"
            ],
            'analysis': [
                "Analyze customer satisfaction data from Q3 surveys",
                "Evaluate performance metrics for marketing campaigns",
                "Review financial statements for investment decision",
                "Assess risk factors in new product launch",
                "Compare pricing strategies across competitors"
            ],
            'development': [
                "Develop automated testing framework for web application",
                "Create data pipeline for real-time analytics",
                "Build recommendation system for e-commerce platform",
                "Implement machine learning model for fraud detection",
                "Design API for mobile application integration"
            ],
            'planning': [
                "Plan quarterly business review presentation",
                "Design onboarding process for new employees",
                "Create project timeline for software migration",
                "Develop crisis communication strategy",
                "Plan user experience improvements for mobile app"
            ]
        }
    
    def _initialize_tool_transitions(self) -> Dict[ToolType, Dict[ToolType, float]]:
        """Initialize realistic tool transition probabilities."""
        return {
            ToolType.WEB_SEARCH: {
                ToolType.READ_DOCUMENT: 0.4,
                ToolType.DEEP_RESEARCH: 0.3,
                ToolType.ANALYZE_DATA: 0.2,
                ToolType.EXTERNAL_API: 0.1
            },
            ToolType.READ_DOCUMENT: {
                ToolType.PREPARE_DOCUMENT: 0.3,
                ToolType.ANALYZE_DATA: 0.25,
                ToolType.MEMORY_STORE_RETRIEVE: 0.25,
                ToolType.WEB_SEARCH: 0.2
            },
            ToolType.ANALYZE_DATA: {
                ToolType.MEMORY_STORE_RETRIEVE: 0.3,
                ToolType.PREPARE_DOCUMENT: 0.25,
                ToolType.WRITE_CODE: 0.25,
                ToolType.WEB_SEARCH: 0.2
            },
            ToolType.DEEP_RESEARCH: {
                ToolType.WEB_SEARCH: 0.3,
                ToolType.READ_DOCUMENT: 0.3,
                ToolType.ANALYZE_DATA: 0.25,
                ToolType.EXTERNAL_API: 0.15
            },
            ToolType.WRITE_CODE: {
                ToolType.ANALYZE_DATA: 0.3,
                ToolType.EXTERNAL_API: 0.3,
                ToolType.MEMORY_STORE_RETRIEVE: 0.2,
                ToolType.PREPARE_DOCUMENT: 0.2
            },
            ToolType.MEMORY_STORE_RETRIEVE: {
                ToolType.ANALYZE_DATA: 0.3,
                ToolType.PREPARE_DOCUMENT: 0.3,
                ToolType.WEB_SEARCH: 0.2,
                ToolType.WRITE_CODE: 0.2
            },
            ToolType.PREPARE_DOCUMENT: {
                ToolType.MEMORY_STORE_RETRIEVE: 0.4,
                ToolType.ANALYZE_DATA: 0.3,
                ToolType.WEB_SEARCH: 0.2,
                ToolType.EXTERNAL_API: 0.1
            },
            ToolType.EXTERNAL_API: {
                ToolType.ANALYZE_DATA: 0.4,
                ToolType.MEMORY_STORE_RETRIEVE: 0.3,
                ToolType.PREPARE_DOCUMENT: 0.2,
                ToolType.WRITE_CODE: 0.1
            }
        }
    
    def generate_trajectories(self, num_trajectories: int) -> List[AgentTrajectory]:
        """
        Generate a set of normal (non-anomalous) trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
        
        Returns:
            List of generated trajectories
        """
        trajectories = []
        
        logger.info("Generating %d normal trajectories", num_trajectories)
        
        with Timer() as timer:
            for i in tqdm(range(num_trajectories), desc="Generating trajectories"):
                # Select pattern based on configured weights
                pattern = self._select_trajectory_pattern()
                
                # Generate trajectory based on pattern
                trajectory = self._generate_trajectory_by_pattern(pattern)
                
                # Add some realistic variance
                self._add_trajectory_variance(trajectory)
                
                # Calculate final metrics
                trajectory.calculate_metrics()
                
                trajectories.append(trajectory)
        
        logger.info("Generated %d trajectories in %.2f seconds", 
                   len(trajectories), timer.elapsed())
        
        return trajectories
    
    def _select_trajectory_pattern(self) -> str:
        """Select trajectory pattern based on configured weights."""
        patterns = list(self.patterns.keys())
        weights = [self.patterns[pattern].get('weight', 1.0) for pattern in patterns]
        
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        return np.random.choice(patterns, p=probabilities)
    
    def _generate_trajectory_by_pattern(self, pattern: str) -> AgentTrajectory:
        """Generate trajectory based on specific pattern."""
        if pattern == "simple_linear":
            return self._generate_simple_linear_trajectory()
        elif pattern == "branched_analysis":
            return self._generate_branched_analysis_trajectory()
        elif pattern == "multi_agent_handoffs":
            return self._generate_multi_agent_handoff_trajectory()
        elif pattern == "complex_research":
            return self._generate_complex_research_trajectory()
        elif pattern == "error_recovery":
            return self._generate_error_recovery_trajectory()
        else:
            # Default to simple linear
            return self._generate_simple_linear_trajectory()
    
    def _generate_simple_linear_trajectory(self) -> AgentTrajectory:
        """Generate a simple linear trajectory with clear step progression (5-10 nodes)."""
        config = self.patterns.get('simple_linear', {})
        num_nodes = random.randint(config.get('min_nodes', 5), config.get('max_nodes', 10))
        
        trajectory = AgentTrajectory(
            task_description=self._get_random_task_description('analysis'),
            start_time=datetime.now()
        )
        
        current_time = trajectory.start_time
        all_nodes = []
        
        # Step 1: User enquiry (clear start node)
        user_node = self._create_user_enquiry_node(current_time)
        all_nodes.append(user_node)
        current_time += timedelta(seconds=random.uniform(1.0, 3.0))
        user_node.end_time = current_time
        user_node.calculate_duration()
        
        # Step 2: Planning phase
        planning_node = self._create_planning_node(current_time)
        all_nodes.append(planning_node)
        current_time += timedelta(seconds=random.uniform(2.0, 5.0))
        planning_node.end_time = current_time
        planning_node.calculate_duration()
        
        # Steps 3-N: Execution phase (remaining nodes - 2 for user and planning)
        execution_nodes = num_nodes - 2
        for i in range(execution_nodes):
            if i == execution_nodes - 1:
                # Final action node (clear end node)
                node = self._create_action_node(current_time)
            else:
                # Intermediate execution nodes
                if random.random() < 0.6:
                    # Tool calls for execution
                    node = self._create_tool_call_node(current_time)
                elif random.random() < 0.5:
                    # LLM calls for reasoning/analysis
                    node = self._create_llm_call_node(current_time, "reasoning")
                else:
                    # Reasoning nodes
                    node = self._create_reasoning_node(current_time)
            
            all_nodes.append(node)
            current_time += timedelta(seconds=random.uniform(1.0, 10.0))
            node.end_time = current_time
            node.calculate_duration()
        
        # Build the trajectory with clear sequential steps
        current_time = self._build_sequential_trajectory(all_nodes, trajectory, trajectory.start_time)
        
        trajectory.end_time = current_time
        
        # Ensure proper connectivity
        self._ensure_trajectory_connectivity(trajectory)
        
        return trajectory
    
    def _ensure_trajectory_connectivity(self, trajectory: AgentTrajectory) -> None:
        """Ensure all nodes in a normal trajectory are properly connected in sequence."""
        if len(trajectory.nodes) < 2:
            return
        
        # Create a set of all node IDs
        all_node_ids = {node.node_id for node in trajectory.nodes}
        
        # Create a set of connected node IDs from edges
        connected_node_ids = set()
        for edge in trajectory.edges:
            connected_node_ids.add(edge.source_node_id)
            connected_node_ids.add(edge.target_node_id)
        
        # Find disconnected nodes
        disconnected_node_ids = all_node_ids - connected_node_ids
        
        if disconnected_node_ids:
            logger.warning(f"Found {len(disconnected_node_ids)} disconnected nodes in normal trajectory, connecting them")
            
            # Create a simple linear chain for disconnected nodes
            disconnected_nodes = [node for node in trajectory.nodes if node.node_id in disconnected_node_ids]
            connected_nodes = [node for node in trajectory.nodes if node.node_id in connected_node_ids]
            
            if connected_nodes and disconnected_nodes:
                # Connect the first disconnected node to the last connected node
                first_disconnected = disconnected_nodes[0]
                last_connected = connected_nodes[-1]
                
                # Use current time if start_time is None
                edge_time = first_disconnected.start_time or datetime.now()
                edge = self._create_execution_edge(last_connected, first_disconnected, edge_time)
                trajectory.edges.append(edge)
                
                # Connect remaining disconnected nodes in sequence
                for i in range(len(disconnected_nodes) - 1):
                    edge_time = disconnected_nodes[i + 1].start_time or datetime.now()
                    edge = self._create_execution_edge(disconnected_nodes[i], disconnected_nodes[i + 1], edge_time)
                    trajectory.edges.append(edge)
            elif disconnected_nodes:
                # If no connected nodes, create a simple chain
                for i in range(len(disconnected_nodes) - 1):
                    edge_time = disconnected_nodes[i + 1].start_time or datetime.now()
                    edge = self._create_execution_edge(disconnected_nodes[i], disconnected_nodes[i + 1], edge_time)
                    trajectory.edges.append(edge)
    
    def _build_sequential_trajectory(self, nodes: List[TrajectoryNode], 
                                   trajectory: AgentTrajectory, 
                                   current_time: datetime) -> datetime:
        """
        Build a trajectory with clear sequential steps from Step 1 to Step N.
        
        Args:
            nodes: List of nodes to add to trajectory
            trajectory: The trajectory to build
            current_time: Current timestamp
            
        Returns:
            Updated current_time
        """
        for i, node in enumerate(nodes):
            # Add node to trajectory
            trajectory.add_node(node)
            
            # Connect to previous node if not the first node
            if i > 0:
                edge = self._create_execution_edge(nodes[i-1], node, current_time)
                trajectory.add_edge(edge)
            
            # Update timing
            if i == 0:
                # First node starts at current_time
                node.start_time = current_time
            else:
                # Subsequent nodes start after previous node ends
                node.start_time = current_time
            
            # Calculate duration and end time
            duration = random.uniform(1.0, 10.0)
            node.end_time = node.start_time + timedelta(seconds=duration)
            node.calculate_duration()
            
            # Update current_time for next node
            current_time = node.end_time
        
        return current_time
    
    def _create_workflow_phase(self, phase_name: str, num_nodes: int, 
                             current_time: datetime, agent_type: AgentType = AgentType.EXECUTOR) -> Tuple[List[TrajectoryNode], datetime]:
        """
        Create a workflow phase with consistent node types and clear progression.
        
        Args:
            phase_name: Name of the phase (e.g., "planning", "execution", "validation")
            num_nodes: Number of nodes in this phase
            current_time: Current timestamp
            agent_type: Agent type for this phase
            
        Returns:
            Tuple of (nodes, updated_current_time)
        """
        nodes = []
        
        for i in range(num_nodes):
            if phase_name == "planning":
                if i == 0:
                    node = self._create_planning_node(current_time, agent_type)
                else:
                    node = self._create_reasoning_node(current_time, agent_type)
            elif phase_name == "execution":
                if random.random() < 0.6:
                    node = self._create_tool_call_node(current_time, agent_type)
                elif random.random() < 0.5:
                    node = self._create_llm_call_node(current_time, "reasoning", agent_type)
                else:
                    node = self._create_reasoning_node(current_time, agent_type)
            elif phase_name == "validation":
                if random.random() < 0.7:
                    node = self._create_validation_node(current_time, agent_type)
                else:
                    node = self._create_llm_call_node(current_time, "validation_analysis", agent_type)
            elif phase_name == "research":
                tools = [ToolType.WEB_SEARCH, ToolType.DEEP_RESEARCH, ToolType.READ_DOCUMENT, ToolType.EXTERNAL_API]
                tool = random.choice(tools)
                node = self._create_tool_call_node(current_time, agent_type, tool_type=tool)
            elif phase_name == "analysis":
                if random.random() < 0.7:
                    node = self._create_tool_call_node(current_time, agent_type, tool_type=ToolType.ANALYZE_DATA)
                else:
                    node = self._create_reasoning_node(current_time, agent_type)
            else:
                # Default to reasoning
                node = self._create_reasoning_node(current_time, agent_type)
            
            nodes.append(node)
            current_time += timedelta(seconds=random.uniform(0.5, 2.0))
        
        return nodes, current_time
    
    def _generate_branched_analysis_trajectory(self) -> AgentTrajectory:
        """Generate a branched analysis trajectory with clear step progression (10-20 nodes)."""
        config = self.patterns.get('branched_analysis', {})
        num_nodes = random.randint(config.get('min_nodes', 10), config.get('max_nodes', 20))
        
        trajectory = AgentTrajectory(
            task_description=self._get_random_task_description('analysis'),
            start_time=datetime.now()
        )
        
        current_time = trajectory.start_time
        all_nodes = []
        
        # Step 1: User enquiry (clear start node)
        user_node = self._create_user_enquiry_node(current_time)
        all_nodes.append(user_node)
        current_time += timedelta(seconds=random.uniform(1.0, 3.0))
        user_node.end_time = current_time
        user_node.calculate_duration()
        
        # Step 2: Initial planning
        planning_node = self._create_planning_node(current_time)
        all_nodes.append(planning_node)
        current_time += timedelta(seconds=random.uniform(2.0, 5.0))
        planning_node.end_time = current_time
        planning_node.calculate_duration()
        
        # Steps 3-N: Create branching structure with clear phases
        branch_points = random.randint(2, 4)
        remaining_nodes = num_nodes - 2  # Subtract user and planning nodes
        
        for branch_idx in range(branch_points):
            branch_size = remaining_nodes // (branch_points - branch_idx)
            remaining_nodes -= branch_size
            
            # Create branch with clear phase progression
            branch_nodes = []
            for i in range(branch_size):
                if i == branch_size - 1:
                    # Final node in branch
                    node = self._create_validation_node(current_time)
                else:
                    # Intermediate nodes using workflow phases
                    if random.random() < 0.6:
                        node = self._create_tool_call_node(current_time)
                    else:
                        node = self._create_reasoning_node(current_time)
                
                branch_nodes.append(node)
                current_time += timedelta(seconds=random.uniform(1.0, 8.0))
                node.end_time = current_time
                node.calculate_duration()
            
            all_nodes.extend(branch_nodes)
        
        # Final step: Action node that consolidates all branches
        final_node = self._create_action_node(current_time)
        all_nodes.append(final_node)
        current_time += timedelta(seconds=random.uniform(2.0, 5.0))
        final_node.end_time = current_time
        final_node.calculate_duration()
        
        # Build the trajectory with clear sequential steps
        current_time = self._build_sequential_trajectory(all_nodes, trajectory, trajectory.start_time)
        
        # Connect branch endpoints to final node
        for node in trajectory.nodes:
            if node.node_type == NodeType.VALIDATION:
                edge = self._create_execution_edge(node, final_node, current_time)
                trajectory.add_edge(edge)
        
        trajectory.end_time = current_time
        return trajectory
    
    def _generate_multi_agent_handoff_trajectory(self) -> AgentTrajectory:
        """Generate a multi-agent handoff trajectory (15-25 nodes)."""
        config = self.patterns.get('multi_agent_handoffs', {})
        num_nodes = random.randint(config.get('min_nodes', 15), config.get('max_nodes', 25))
        
        trajectory = AgentTrajectory(
            task_description=self._get_random_task_description('development'),
            start_time=datetime.now()
        )
        
        current_time = trajectory.start_time
        nodes = []
        
        # Start with user enquiry (clear start node)
        user_node = self._create_user_enquiry_node(current_time)
        trajectory.add_node(user_node)
        nodes.append(user_node)
        current_time += timedelta(seconds=random.uniform(1.0, 3.0))
        user_node.end_time = current_time
        user_node.calculate_duration()
        
        # Initial planning by coordinator
        planning_node = self._create_planning_node(current_time, AgentType.COORDINATOR)
        trajectory.add_node(planning_node)
        nodes.append(planning_node)
        
        # Connect user enquiry to planning
        edge = self._create_execution_edge(user_node, planning_node, current_time)
        trajectory.add_edge(edge)
        
        current_time += timedelta(seconds=random.uniform(2.0, 5.0))
        planning_node.end_time = current_time
        planning_node.calculate_duration()
        
        # Multi-agent execution phase
        agents = [AgentType.PLANNER, AgentType.EXECUTOR, AgentType.VALIDATOR]
        current_agent_idx = 0
        remaining_nodes = num_nodes - 2  # Subtract user and planning nodes
        
        for i in range(remaining_nodes):
            current_agent = agents[current_agent_idx]
            
            # Decide if we should hand off to another agent (but not too frequently)
            should_handoff = (i > 0 and 
                            random.random() < 0.2 and  # Reduced handoff frequency
                            i < remaining_nodes - 1 and
                            len(nodes) > 3)  # Ensure we have some work done
            
            if should_handoff:
                # Create handoff node
                source_agent = agents[current_agent_idx]
                current_agent_idx = (current_agent_idx + 1) % len(agents)
                target_agent = agents[current_agent_idx]
                
                handoff_node = self._create_handoff_node(current_time, source_agent, target_agent)
                trajectory.add_node(handoff_node)
                nodes.append(handoff_node)
                
                # Connect to previous node
                edge = self._create_handoff_edge(nodes[-2], handoff_node, current_time)
                trajectory.add_edge(edge)
                
                current_time += timedelta(seconds=random.uniform(0.5, 2.0))
                handoff_node.end_time = current_time
                handoff_node.calculate_duration()
                
                # Update current agent
                current_agent = target_agent
            else:
                # Create regular node based on current agent
                if current_agent == AgentType.PLANNER:
                    if random.random() < 0.7:
                        node = self._create_llm_call_node(current_time, "planning", current_agent)
                    else:
                        node = self._create_reasoning_node(current_time, current_agent)
                elif current_agent == AgentType.EXECUTOR:
                    if random.random() < 0.6:
                        node = self._create_tool_call_node(current_time, current_agent)
                    elif random.random() < 0.5:
                        node = self._create_llm_call_node(current_time, "reasoning", current_agent)
                    else:
                        node = self._create_reasoning_node(current_time, current_agent)
                else:  # VALIDATOR
                    if random.random() < 0.6:
                        node = self._create_validation_node(current_time, current_agent)
                    else:
                        node = self._create_llm_call_node(current_time, "validation_analysis", current_agent)
                
                trajectory.add_node(node)
                nodes.append(node)
                
                # Connect to previous node
                edge = self._create_execution_edge(nodes[-2], node, current_time)
                trajectory.add_edge(edge)
            
            current_time += timedelta(seconds=random.uniform(0.5, 15.0))
            if not should_handoff:
                node.end_time = current_time
                node.calculate_duration()
        
        # Final action node (clear end node)
        final_node = self._create_action_node(current_time, AgentType.EXECUTOR)
        trajectory.add_node(final_node)
        nodes.append(final_node)
        
        # Connect to last node
        edge = self._create_execution_edge(nodes[-2], final_node, current_time)
        trajectory.add_edge(edge)
        
        current_time += timedelta(seconds=random.uniform(2.0, 5.0))
        final_node.end_time = current_time
        final_node.calculate_duration()
        
        trajectory.end_time = current_time
        return trajectory
    
    def _generate_complex_research_trajectory(self) -> AgentTrajectory:
        """Generate a complex research trajectory (20-40 nodes)."""
        config = self.patterns.get('complex_research', {})
        num_nodes = random.randint(config.get('min_nodes', 20), config.get('max_nodes', 40))
        
        trajectory = AgentTrajectory(
            task_description=self._get_random_task_description('research'),
            start_time=datetime.now()
        )
        
        current_time = trajectory.start_time
        nodes = []
        
        # Start with user enquiry (clear start node)
        user_node = self._create_user_enquiry_node(current_time)
        trajectory.add_node(user_node)
        nodes.append(user_node)
        current_time += timedelta(seconds=random.uniform(1.0, 3.0))
        user_node.end_time = current_time
        user_node.calculate_duration()
        
        # Phase 1: Initial research planning (20% of nodes)
        planning_nodes = max(2, int(num_nodes * 0.2))
        for i in range(planning_nodes):
            if i == 0:
                node = self._create_planning_node(current_time)
            else:
                node = self._create_reasoning_node(current_time)
            
            trajectory.add_node(node)
            nodes.append(node)
            
            # Connect to previous node or user enquiry
            if i == 0:
                edge = self._create_execution_edge(user_node, node, current_time)
            else:
                edge = self._create_execution_edge(nodes[-2], node, current_time)
            trajectory.add_edge(edge)
            
            current_time += timedelta(seconds=random.uniform(2.0, 8.0))
            node.end_time = current_time
            node.calculate_duration()
        
        # Phase 2: Data gathering (40% of nodes)
        gathering_nodes = max(3, int(num_nodes * 0.4))
        for i in range(gathering_nodes):
            # Alternate between different research tools
            tools = [ToolType.WEB_SEARCH, ToolType.DEEP_RESEARCH, ToolType.READ_DOCUMENT, ToolType.EXTERNAL_API]
            tool = random.choice(tools)
            
            node = self._create_tool_call_node(current_time, tool_type=tool)
            trajectory.add_node(node)
            
            # Connect to last planning node or previous gathering node
            edge = self._create_execution_edge(nodes[-1], node, current_time)
            trajectory.add_edge(edge)
            nodes.append(node)
            
            # Add memory storage after data gathering
            if random.random() < 0.6:
                memory_node = self._create_memory_node(current_time + timedelta(seconds=1))
                trajectory.add_node(memory_node)
                memory_edge = self._create_execution_edge(node, memory_node, current_time)
                trajectory.add_edge(memory_edge)
                
                current_time += timedelta(seconds=random.uniform(0.5, 2.0))
                memory_node.end_time = current_time
                memory_node.calculate_duration()
                nodes.append(memory_node)
            
            current_time += timedelta(seconds=random.uniform(3.0, 12.0))
            node.end_time = current_time
            node.calculate_duration()
        
        # Phase 3: Analysis (30% of nodes)
        analysis_nodes = max(2, int(num_nodes * 0.3))
        for i in range(analysis_nodes):
            if random.random() < 0.7:
                node = self._create_tool_call_node(current_time, tool_type=ToolType.ANALYZE_DATA)
            else:
                node = self._create_reasoning_node(current_time)
            
            trajectory.add_node(node)
            
            # Connect to previous node
            edge = self._create_execution_edge(nodes[-1], node, current_time)
            trajectory.add_edge(edge)
            nodes.append(node)
            
            current_time += timedelta(seconds=random.uniform(5.0, 20.0))
            node.end_time = current_time
            node.calculate_duration()
        
        # Phase 4: Validation and final output (10% of nodes)
        final_nodes = max(1, int(num_nodes * 0.1))
        for i in range(final_nodes):
            if i == final_nodes - 1:
                node = self._create_action_node(current_time)  # Clear end node
            else:
                node = self._create_validation_node(current_time)
            
            trajectory.add_node(node)
            
            edge = self._create_execution_edge(nodes[-1], node, current_time)
            trajectory.add_edge(edge)
            nodes.append(node)
            
            current_time += timedelta(seconds=random.uniform(2.0, 8.0))
            node.end_time = current_time
            node.calculate_duration()
        
        trajectory.end_time = current_time
        return trajectory
    
    def _generate_error_recovery_trajectory(self) -> AgentTrajectory:
        """Generate an error recovery trajectory (10-30 nodes)."""
        config = self.patterns.get('error_recovery', {})
        num_nodes = random.randint(config.get('min_nodes', 10), config.get('max_nodes', 30))
        
        trajectory = AgentTrajectory(
            task_description=self._get_random_task_description('development'),
            start_time=datetime.now()
        )
        
        current_time = trajectory.start_time
        previous_node = None
        error_injected = False
        
        for i in range(num_nodes):
            # Inject an error in the middle part of the trajectory
            should_inject_error = (not error_injected and 
                                 i > num_nodes * 0.3 and 
                                 i < num_nodes * 0.7 and 
                                 random.random() < 0.4)
            
            if should_inject_error:
                # Create a failing tool call
                node = self._create_tool_call_node(current_time, success=False)
                error_injected = True
                
                # Follow with recovery nodes
                next_nodes_count = min(3, num_nodes - i - 1)
                trajectory.add_node(node)
                
                if previous_node:
                    edge = self._create_execution_edge(previous_node, node, current_time)
                    trajectory.add_edge(edge)
                
                current_time += timedelta(seconds=random.uniform(2.0, 5.0))
                node.end_time = current_time
                node.calculate_duration()
                
                # Add recovery sequence
                recovery_nodes = []
                for j in range(next_nodes_count):
                    if j == 0:
                        # Error observation
                        recovery_node = self._create_observation_node(current_time)
                    elif j == 1:
                        # Retry or alternative approach
                        recovery_node = self._create_tool_call_node(current_time, success=True)
                        recovery_node.retry_count = 1
                    else:
                        # Validation of recovery
                        recovery_node = self._create_validation_node(current_time)
                    
                    trajectory.add_node(recovery_node)
                    recovery_nodes.append(recovery_node)
                    
                    # Connect recovery nodes
                    if j == 0:
                        edge = self._create_execution_edge(node, recovery_node, current_time)
                    else:
                        edge = self._create_execution_edge(recovery_nodes[j-1], recovery_node, current_time)
                    
                    trajectory.add_edge(edge)
                    
                    current_time += timedelta(seconds=random.uniform(1.0, 8.0))
                    recovery_node.end_time = current_time
                    recovery_node.calculate_duration()
                
                previous_node = recovery_nodes[-1] if recovery_nodes else node
                i += next_nodes_count  # Skip ahead
                
            else:
                # Create normal node
                if i == 0:
                    node = self._create_user_enquiry_node(current_time)  # Clear start node
                elif i == num_nodes - 1:
                    node = self._create_action_node(current_time)  # Clear end node
                else:
                    if i == 1:
                        # Second node should be planning
                        node = self._create_planning_node(current_time)
                    elif random.random() < 0.6:
                        node = self._create_tool_call_node(current_time)
                    else:
                        node = self._create_reasoning_node(current_time)
                
                trajectory.add_node(node)
                
                if previous_node:
                    edge = self._create_execution_edge(previous_node, node, current_time)
                    trajectory.add_edge(edge)
                
                current_time += timedelta(seconds=random.uniform(1.0, 10.0))
                node.end_time = current_time
                node.calculate_duration()
                
                previous_node = node
        
        trajectory.end_time = current_time
        return trajectory
    
    def _create_planning_node(self, start_time: datetime, agent_type: AgentType = AgentType.PLANNER) -> TrajectoryNode:
        """Create a planning node."""
        planning_types = ["task_decomposition", "strategy_selection", "resource_allocation"]
        
        return TrajectoryNode(
            node_type=NodeType.PLANNING,
            agent_type=agent_type,
            start_time=start_time,
            description=f"Planning phase: {random.choice(planning_types)}",
            planning_type=random.choice(planning_types),
            planning_confidence=random.uniform(0.7, 0.95),
            planning_output={
                "strategy": f"strategy_{random.randint(1, 10)}",
                "estimated_duration": random.randint(300, 1800),
                "resources_needed": random.randint(1, 5)
            }
        )
    
    def _create_user_enquiry_node(self, start_time: datetime) -> TrajectoryNode:
        """Create a user enquiry node."""
        enquiry_templates = [
            "Can you analyze the quarterly sales data?",
            "Help me understand the customer feedback trends",
            "What are the key insights from our latest marketing campaign?",
            "Generate a report on competitor analysis",
            "Create a summary of the project status",
            "What are the potential risks in our current strategy?",
            "Help me optimize the website performance",
            "Analyze the effectiveness of our pricing model"
        ]
        
        user_input = random.choice(enquiry_templates)
        return TrajectoryNode(
            node_type=NodeType.USER_ENQUIRY,
            agent_type=AgentType.COORDINATOR,  # Coordinator handles user interactions
            start_time=start_time,
            description=f"User enquiry: {user_input[:50]}...",
            user_input=user_input,
            user_intent="question",
            user_context={
                "user_id": f"user_{random.randint(1, 1000)}",
                "session_id": f"session_{random.randint(1, 100)}",
                "priority": random.choice(["low", "medium", "high"])
            },
            user_satisfaction=None  # Will be set after response
        )
    
    def _create_llm_call_node(self, start_time: datetime, purpose: str, agent_type: AgentType = AgentType.EXECUTOR) -> TrajectoryNode:
        """Create an LLM call node."""
        llm_models = ["gpt-4", "claude-3", "gemini-pro", "llama-2"]
        model = random.choice(llm_models)
        
        # Generate appropriate prompts based on purpose
        prompt_templates = {
            "planning": [
                "Analyze the user request and create a step-by-step plan",
                "Break down the task into manageable components",
                "Identify the key requirements and constraints",
                "Create an execution strategy for this task"
            ],
            "reasoning": [
                "Analyze the data and provide insights",
                "Evaluate the current situation and identify patterns",
                "Consider multiple perspectives and draw conclusions",
                "Apply logical reasoning to solve this problem"
            ],
            "response_generation": [
                "Generate a comprehensive response to the user",
                "Create a clear and actionable summary",
                "Formulate recommendations based on the analysis",
                "Prepare a detailed report for the user"
            ],
            "analysis": [
                "Perform deep analysis of the provided information",
                "Identify trends and correlations in the data",
                "Evaluate the quality and reliability of sources",
                "Extract key insights from the research findings"
            ]
        }
        
        prompt = random.choice(prompt_templates.get(purpose, ["Process the request"]))
        
        return TrajectoryNode(
            node_type=NodeType.LLM_CALL,
            agent_type=agent_type,
            start_time=start_time,
            description=f"LLM {purpose}: {prompt[:40]}...",
            llm_model=model,
            llm_prompt=prompt,
            llm_purpose=purpose,
            llm_tokens_used=random.randint(100, 2000),
            llm_confidence=random.uniform(0.7, 0.95),
            llm_response=f"Generated response for {purpose} using {model}"
        )
    
    def _create_tool_call_node(self, start_time: datetime, agent_type: AgentType = AgentType.EXECUTOR, 
                              tool_type: Optional[ToolType] = None, success: bool = True) -> TrajectoryNode:
        """Create a tool call node."""
        if tool_type is None:
            tool_type = random.choice(list(ToolType))
        
        tool_params = self._generate_tool_parameters(tool_type)
        node = TrajectoryNode(
            node_type=NodeType.TOOL_CALL,
            agent_type=agent_type,
            start_time=start_time,
            description=f"Execute {tool_type.value}: {str(tool_params)[:30]}...",
            tool_type=tool_type,
            tool_success=success,
            tool_parameters=tool_params,
            tool_result=self._generate_tool_result(tool_type, success),
            network_calls=random.randint(1, 5) if tool_type in [ToolType.WEB_SEARCH, ToolType.EXTERNAL_API] else 0
        )
        
        if not success:
            node.status = "failed"
            node.error_code = f"ERR_{random.randint(1000, 9999)}"
            node.error_message = f"Tool {tool_type.value} failed to execute"
            node.tool_error_message = node.error_message
        
        return node
    
    def _create_reasoning_node(self, start_time: datetime, agent_type: AgentType = AgentType.EXECUTOR) -> TrajectoryNode:
        """Create a reasoning node."""
        reasoning_types = ["analysis", "inference", "decision_making"]
        reasoning_type = random.choice(reasoning_types)
        data_points = random.randint(5, 50)
        
        return TrajectoryNode(
            node_type=NodeType.REASONING,
            agent_type=agent_type,
            start_time=start_time,
            description=f"Reasoning {reasoning_type}: analyze {data_points} data points",
            reasoning_type=reasoning_type,
            reasoning_confidence=random.uniform(0.6, 0.9),
            reasoning_input={"data_points": data_points},
            reasoning_output={"conclusion": f"result_{random.randint(1, 100)}"}
        )
    
    def _create_validation_node(self, start_time: datetime, agent_type: AgentType = AgentType.VALIDATOR) -> TrajectoryNode:
        """Create a validation node."""
        criteria = ["accuracy", "completeness", "relevance", "consistency"]
        
        return TrajectoryNode(
            node_type=NodeType.VALIDATION,
            agent_type=agent_type,
            start_time=start_time,
            description="Validate results and quality",
            validation_criteria=random.sample(criteria, random.randint(1, 3)),
            validation_result=random.random() > 0.1,  # 90% validation success rate
            validation_score=random.uniform(0.7, 0.95)
        )
    
    def _create_handoff_node(self, start_time: datetime, source_agent: AgentType, target_agent: AgentType) -> TrajectoryNode:
        """Create a handoff node."""
        return TrajectoryNode(
            node_type=NodeType.HANDOFF,
            agent_type=AgentType.COORDINATOR,
            start_time=start_time,
            description=f"Handoff from {source_agent.value} to {target_agent.value}",
            source_agent=source_agent,
            target_agent=target_agent,
            handoff_success=random.random() > 0.05,  # 95% handoff success rate
            handoff_context={
                "context_size": random.randint(100, 1000),
                "data_transferred": random.randint(1, 10)
            }
        )
    
    def _create_memory_node(self, start_time: datetime, agent_type: AgentType = AgentType.EXECUTOR) -> TrajectoryNode:
        """Create a memory access node."""
        operations = ["store", "retrieve", "update", "delete"]
        operation = random.choice(operations)
        
        return TrajectoryNode(
            node_type=NodeType.MEMORY_ACCESS,
            agent_type=agent_type,
            start_time=start_time,
            description=f"Memory {operation}",
            memory_operation=operation,
            memory_key=f"key_{random.randint(1, 1000)}",
            memory_success=random.random() > 0.05,  # 95% memory success rate
            memory_value={"size": random.randint(10, 1000)} if operation in ["store", "update"] else None
        )
    
    def _create_observation_node(self, start_time: datetime, agent_type: AgentType = AgentType.COORDINATOR) -> TrajectoryNode:
        """Create an observation node."""
        sources = ["system_metrics", "user_feedback", "performance_data", "error_logs"]
        
        return TrajectoryNode(
            node_type=NodeType.OBSERVATION,
            agent_type=agent_type,
            start_time=start_time,
            description="Observe system state",
            observation_source=random.choice(sources),
            observation_data={
                "metrics_count": random.randint(5, 20),
                "anomalies_detected": random.randint(0, 3)
            }
        )
    
    def _create_action_node(self, start_time: datetime, agent_type: AgentType = AgentType.EXECUTOR) -> TrajectoryNode:
        """Create a final action node."""
        return TrajectoryNode(
            node_type=NodeType.ACTION,
            agent_type=agent_type,
            start_time=start_time,
            description="Execute final action",
            content={"final_output": f"result_{random.randint(1, 1000)}"}
        )
    
    def _create_execution_edge(self, source: TrajectoryNode, target: TrajectoryNode, creation_time: datetime) -> TrajectoryEdge:
        """Create a standard execution flow edge."""
        return TrajectoryEdge(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type="execution_flow",
            relationship_type="sequential",
            creation_time=creation_time,
            latency=random.uniform(0.1, 2.0),
            probability=random.uniform(0.8, 1.0),
            confidence=random.uniform(0.85, 0.98)
        )
    
    def _create_handoff_edge(self, source: TrajectoryNode, target: TrajectoryNode, creation_time: datetime) -> TrajectoryEdge:
        """Create a handoff edge."""
        return TrajectoryEdge(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type="handoff",
            relationship_type="sequential",
            creation_time=creation_time,
            latency=random.uniform(0.5, 5.0),
            probability=random.uniform(0.9, 1.0),
            confidence=random.uniform(0.8, 0.95),
            data_transferred={"context_data": True}
        )
    
    def _create_data_dependency_edge(self, source: TrajectoryNode, target: TrajectoryNode, creation_time: datetime) -> TrajectoryEdge:
        """Create a data dependency edge."""
        return TrajectoryEdge(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type="data_dependency",
            relationship_type="dependency",
            creation_time=creation_time,
            latency=random.uniform(0.1, 1.0),
            probability=random.uniform(0.9, 1.0),
            confidence=random.uniform(0.9, 0.98),
            data_transferred={"data_size": random.randint(100, 10000)}
        )
    
    def _generate_tool_parameters(self, tool_type: ToolType) -> Dict:
        """Generate realistic tool parameters."""
        base_params = {
            ToolType.WEB_SEARCH: {"query": f"search_term_{random.randint(1, 1000)}", "max_results": random.randint(5, 20)},
            ToolType.READ_DOCUMENT: {"document_id": f"doc_{random.randint(1, 1000)}", "page_range": "1-10"},
            ToolType.ANALYZE_DATA: {"dataset": f"data_{random.randint(1, 100)}", "analysis_type": "statistical"},
            ToolType.EXTERNAL_API: {"endpoint": f"api_endpoint_{random.randint(1, 10)}", "timeout": 30}
        }
        return base_params.get(tool_type, {"param": f"value_{random.randint(1, 100)}"})
    
    def _generate_tool_result(self, tool_type: ToolType, success: bool) -> Dict:
        """Generate realistic tool results."""
        if not success:
            return {"error": "Tool execution failed"}
        
        base_results = {
            ToolType.WEB_SEARCH: {"results_count": random.randint(1, 20), "relevance_score": random.uniform(0.6, 0.9)},
            ToolType.READ_DOCUMENT: {"pages_read": random.randint(1, 10), "content_extracted": True},
            ToolType.ANALYZE_DATA: {"insights": random.randint(3, 15), "confidence": random.uniform(0.7, 0.95)},
            ToolType.EXTERNAL_API: {"status_code": 200, "response_size": random.randint(100, 5000)}
        }
        return base_results.get(tool_type, {"status": "completed"})
    
    def _get_random_task_description(self, category: str = None) -> str:
        """Get a random task description."""
        if category and category in self.task_templates:
            return random.choice(self.task_templates[category])
        
        # Random category if not specified
        all_tasks = []
        for tasks in self.task_templates.values():
            all_tasks.extend(tasks)
        
        # Ensure we always return a string, even if task_templates is empty
        if all_tasks:
            return random.choice(all_tasks)
        else:
            return "Analyze data and generate insights"
    
    def _add_trajectory_variance(self, trajectory: AgentTrajectory) -> None:
        """Add realistic variance to trajectory."""
        # Add random performance metrics
        for node in trajectory.nodes:
            node.cpu_usage = random.uniform(10.0, 80.0)
            node.memory_usage = random.uniform(50.0, 500.0)
            
            # Add some realistic timing variance
            if node.duration:
                variance = random.uniform(0.8, 1.2)
                node.duration *= variance
        
        # Add random edge latencies
        for edge in trajectory.edges:
            edge.latency = random.uniform(0.1, 5.0)
        
        # Add some tags
        tags = ["priority_high", "batch_process", "user_initiated", "automated", "experimental"]
        trajectory.tags.update(random.sample(tags, random.randint(1, 3)))
