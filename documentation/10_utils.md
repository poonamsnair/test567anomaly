# Module: utils.py

## Overview
This module provides the core data structures and utility functions for the agent trajectory anomaly detection system. It defines the fundamental building blocks for representing, manipulating, and analyzing agent trajectories, nodes, and edges.

## Design and Implementation
### 1. Core Data Structures
- **TrajectoryNode:** Represents a single step in an agent's execution (tool call, reasoning, planning, validation, handoff, etc.). Includes detailed attributes for timing, content, tool/memory/planning/validation info, performance, status, errors, anomaly metadata, and more.
- **TrajectoryEdge:** Represents a directed connection between nodes, with attributes for type, relationship, latency, probability, confidence, error counts, reliability, and anomaly status.
- **AgentTrajectory:** Represents a full trajectory as a sequence of nodes and edges, with methods for adding nodes/edges, calculating metrics, and querying structure.
- **Enums:** NodeType, AgentType, ToolType, AnomalyType, SeverityLevel—define all possible types for nodes, agents, tools, anomalies, and severity.

### 2. Utility Functions
- **Timer:** Context manager for timing code execution.
- **Logging Setup:** Configures logging for the application.
- **Config Loading:** Loads YAML configuration files.
- **Pickle Save/Load:** Serializes and deserializes objects for reproducibility.
- **Directory Management:** Ensures output directories exist.
- **Memory Usage:** Reports current memory usage.
- **Duration Formatting:** Converts seconds to human-readable strings.
- **Graph Hashing:** Computes a hash for a graph structure for comparison or deduplication.

## Key Features
- **Comprehensive Data Modeling:** Captures all relevant aspects of agent execution, including user interaction, tool/memory/LLM/planning/validation details, and anomaly metadata.
- **Extensibility:** Data classes and enums are designed for easy extension as new node/edge types or attributes are needed.
- **Robustness:** Handles missing data, error states, and ensures all nodes/edges are valid and connected.
- **Reproducibility:** Utility functions support saving/loading of data and models for experiment tracking.

## Limitations and Concerns
- **Complexity:** The richness of the data model may increase memory usage and serialization time for large datasets.
- **Dependency on Consistency:** Downstream modules rely on consistent use of node/edge types and attributes.
- **Error Handling:** Some utility functions may fail silently if dependencies (e.g., psutil) are missing.

## Design Decisions
- **DataClass Usage:** Uses Python dataclasses for clarity, type safety, and default value management.
- **Enum Types:** Enums ensure only valid types are used throughout the system.
- **Separation of Concerns:** Utility functions are kept separate from core data structures for modularity.

## Outcome
This module provides the foundational data structures and utilities for the entire anomaly detection pipeline, ensuring consistency, extensibility, and reproducibility across all components. 