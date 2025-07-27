# universal_trace_schema.py
"""
Universal Schema for representing agent interaction traces from different sources.
This module defines dataclasses for a standardized representation of agent traces
and provides logic to parse various JSON formats into this schema.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
TRANSFORMER_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
logger = logging.getLogger(__name__)

# --- Core Data Classes ---
@dataclass
class UniversalNode:
    """Represents a single interaction step within a trace."""
    content: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation and normalization."""
        if not isinstance(self.content, str):
            self.content = str(self.content)
        if not isinstance(self.node_type, str):
            self.node_type = str(self.node_type).lower()
        if not isinstance(self.attributes, dict):
            self.attributes = {}

@dataclass
class UniversalEdge:
    """Represents a relationship between two interaction steps."""
    source: int
    target: int
    edge_type: str

    def __post_init__(self):
        """Validation."""
        if not isinstance(self.source, int) or self.source < 0:
             raise ValueError(f"Edge source must be a non-negative integer, got {self.source}")
        if not isinstance(self.target, int) or self.target < 0:
             raise ValueError(f"Edge target must be a non-negative integer, got {self.target}")
        if not isinstance(self.edge_type, str):
             self.edge_type = str(self.edge_type)

# --- Universal Trace Schema ---
@dataclass
class UniversalTraceSchema:
    """
    A standardized representation of an agent interaction trace.
    """
    initial_input: Optional[str] = None
    final_output: Optional[str] = None
    nodes: List[UniversalNode] = field(default_factory=list)
    edges: List[UniversalEdge] = field(default_factory=list)
    resources_used: List[str] = field(default_factory=list)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation and normalization."""
        # Ensure lists and dicts are initialized correctly
        if not isinstance(self.nodes, list):
            self.nodes = []
        if not isinstance(self.edges, list):
            self.edges = []
        if not isinstance(self.resources_used, list):
            self.resources_used = []
        if not isinstance(self.additional_metadata, dict):
            self.additional_metadata = {}

        # Validate and convert nodes
        validated_nodes = []
        for node in self.nodes:
            if isinstance(node, UniversalNode):
                validated_nodes.append(node)
            elif isinstance(node, dict):
                try:
                    validated_nodes.append(UniversalNode(**node))
                except Exception as e:
                    logger.warning(f"Failed to create UniversalNode from dict: {e}")
        self.nodes = validated_nodes

        # Validate and convert edges
        validated_edges = []
        for edge in self.edges:
            if isinstance(edge, UniversalEdge):
                if edge.source < len(self.nodes) and edge.target < len(self.nodes):
                    validated_edges.append(edge)
                else:
                    logger.warning(f"Edge has source/target index out of bounds. Skipping.")
            elif isinstance(edge, dict):
                try:
                    edge_obj = UniversalEdge(**edge)
                    if edge_obj.source < len(self.nodes) and edge_obj.target < len(self.nodes):
                        validated_edges.append(edge_obj)
                    else:
                        logger.warning(f"Edge created from dict has source/target index out of bounds. Skipping.")
                except Exception as e:
                    logger.warning(f"Failed to create UniversalEdge from dict: {e}")
        self.edges = validated_edges

        # Normalize resource names
        self.resources_used = [str(r).lower() for r in self.resources_used if r]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['UniversalTraceSchema']:
        """
        Factory method to create a UniversalTraceSchema from a raw data dictionary.
        """
        if not isinstance(data, dict):
            logger.warning("Input data is not a dictionary.")
            return None

        try:
            # --- 1. Extract Common High-Level Metadata ---
            structural_keys = {
                'user_question', 'initial_input',
                'final_answer', 'final_output',
                'steps', 'messages',
                'tools_used', 'resources_used',
                'errors', 'agents_called', 'agent_actions',
                'metadata'
            }

            # Safely extract initial/final inputs
            initial_input_raw = data.get('user_question') or data.get('initial_input')
            initial_input = str(initial_input_raw) if initial_input_raw is not None else None

            final_output_raw = data.get('final_answer') or data.get('final_output')
            final_output = str(final_output_raw) if final_output_raw is not None else None

            # Extract resources used
            raw_resources = data.get('tools_used', [])
            if not isinstance(raw_resources, list):
                raw_resources = []
            resources_used = [str(r).lower() for r in raw_resources if r]

            # Extract flat additional metadata, excluding structural keys and 'errors' to prevent data leakage
            additional_metadata = {
                k: v for k, v in data.items()
                if k not in structural_keys
            }

            # Handle potential nested 'metadata' key
            nested_metadata = data.get('metadata')
            if isinstance(nested_metadata, dict):
                 additional_metadata.update(nested_metadata)

            # --- 2. Determine Format and Parse Nodes/Edges ---
            nodes: List[UniversalNode] = []
            edges: List[UniversalEdge] = []

            if 'messages' in data and isinstance(data['messages'], list):
                logger.debug("Detected 'messages' format.")
                nodes, edges = cls._parse_messages_format(data)
            elif 'steps' in data:
                logger.debug("Detected 'steps' format.")
                nodes, edges = cls._parse_steps_format(data)
            else:
                logger.warning("Unknown data format. Attempting generic parse.")
                nodes = [UniversalNode(content=json.dumps(data, sort_keys=True), node_type="unknown")]

            # --- 3. Create and Return Schema Object ---
            schema = cls(
                initial_input=initial_input,
                final_output=final_output,
                nodes=nodes,
                edges=edges,
                resources_used=resources_used,
                additional_metadata=additional_metadata
            )

            logger.debug(f"Successfully created UniversalTraceSchema with {len(schema.nodes)} nodes and {len(schema.edges)} edges.")
            return schema

        except Exception as e:
            logger.error(f"Error creating UniversalTraceSchema from data: {e}", exc_info=True)
            return None

    @classmethod
    def _parse_steps_format(cls, data: Dict[str, Any]) -> tuple[List[UniversalNode], List[UniversalEdge]]:
        """Parses the 'steps'/'agent_actions' format (internlm_agent, WebShop)."""
        nodes: List[UniversalNode] = []
        edges: List[UniversalEdge] = []

        type_mapping = {
            'human': 'user', 'user': 'user',
            'ai': 'agent', 'gpt': 'agent', 'llm': 'agent',
            'tool': 'tool', 'observation': 'observation', 'tool_response': 'tool',
            'plan': 'plan', 'answer': 'final_action', 'final': 'final_action',
            'action': 'llm_call', 'llm_call': 'llm_call'
        }

        steps = data.get('steps', [])
        if not isinstance(steps, list):
            steps = []

        for s in steps:
            if not isinstance(s, dict):
                continue
            try:
                content = str(s.get('content', ''))
                raw_type = str(s.get('type', 'unknown')).lower()
                node_type = type_mapping.get(raw_type, 'observation')
                attributes = s.get('additional_kwargs', {})
                if not isinstance(attributes, dict):
                    attributes = {}
                nodes.append(UniversalNode(content=content, node_type=node_type, attributes=attributes))
            except Exception as e:
                logger.error(f"Error processing step: {e}")
                continue

        # Create sequential edges
        for i in range(1, len(nodes)):
            try:
                prev_type = nodes[i-1].node_type
                curr_type = nodes[i].node_type
                edge_type = 'next_step'

                if prev_type == 'agent' and curr_type == 'tool':
                    edge_type = 'tool_call'
                elif prev_type == 'user' and curr_type == 'agent':
                    edge_type = 'handoff'
                elif prev_type in ['agent', 'llm_call'] and curr_type in ['llm_call', 'agent']:
                    edge_type = 'reasoning'
                elif prev_type == curr_type:
                    edge_type = 'loop'
                elif 'error' in nodes[i].content.lower():
                    edge_type = 'error_signal'

                edges.append(UniversalEdge(source=i-1, target=i, edge_type=edge_type))
            except Exception as e:
                 logger.error(f"Error creating edge: {e}")
                 continue

        # Handle agent_actions (common in WebShop and internlm)
        agent_actions = data.get('agent_actions', [])
        if not isinstance(agent_actions, list):
            agent_actions = []

        for a in agent_actions:
            if isinstance(a, dict) and nodes:
                try:
                    last_idx = len(nodes) - 1
                    action_type = str(a.get('type', 'tool')).lower()
                    node_type = 'tool' if action_type in ['tool_use', 'tool'] else 'llm_call'
                    input_data = a.get('input', '')

                    if isinstance(input_data, dict):
                        content = json.dumps(input_data, sort_keys=True)
                    else:
                        content = str(input_data)

                    attributes = {k: v for k, v in a.items() if k not in ['type', 'input']}
                    try:
                        json.dumps(attributes)
                    except (TypeError, ValueError):
                        attributes = {}

                    nodes.append(UniversalNode(content=content, node_type=node_type, attributes=attributes))
                    edges.append(UniversalEdge(source=last_idx, target=len(nodes)-1, edge_type='tool_call' if node_type == 'tool' else 'reasoning'))
                except Exception as e:
                    logger.error(f"Error processing agent action: {e}")
                    continue
            elif isinstance(a, str) and nodes: # Handle WebShop string actions like "Action: click[...]"
                 try:
                     last_idx = len(nodes) - 1
                     # Heuristic: if it looks like an action command, treat it as a tool call node
                     if a.startswith(("Action:", "click[", "search[", "think[")):
                         node_type = 'tool'
                         content = a
                         attributes = {"source_action_string": True} # Flag for post-processing if needed
                         nodes.append(UniversalNode(content=content, node_type=node_type, attributes=attributes))
                         edges.append(UniversalEdge(source=last_idx, target=len(nodes)-1, edge_type='tool_call'))
                     else:
                         # If it's a string but doesn't look like an action, maybe add as a generic observation?
                         # For now, we'll skip non-action strings in the list to be conservative.
                         logger.debug(f"Skipping non-action string in agent_actions: {a[:30]}...")
                 except Exception as e:
                     logger.error(f"Error processing string agent action: {e}")
                     continue

        return nodes, edges

    @classmethod
    def _parse_messages_format(cls, data: Dict[str, Any]) -> tuple[List[UniversalNode], List[UniversalEdge]]:
        """Parses the 'messages' format (snorkelai_finance)."""
        nodes: List[UniversalNode] = []
        edges: List[UniversalEdge] = []

        type_mapping = {
            'human': 'user', 'user': 'user',
            'ai': 'agent', 'assistant': 'agent', 'gpt': 'agent',
            'tool': 'tool', 'tool_response': 'tool', 'observation': 'tool',
            'system': 'system'
        }

        messages = data.get('messages', [])
        if not isinstance(messages, list):
            messages = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            try:
                msg_type = str(msg.get('type', 'unknown')).lower()
                node_type = type_mapping.get(msg_type, 'observation')
                content = ""
                attributes = msg.get('additional_kwargs', {}) or msg.get('kwargs', {})
                if not isinstance(attributes, dict):
                    attributes = {}

                # Content Extraction based on message type
                if msg_type in ['human', 'user']:
                    content = str(msg.get('content', ''))
                elif msg_type in ['ai', 'assistant']:
                    ai_content = msg.get('content', '')
                    if isinstance(ai_content, list):
                        content_parts = []
                        for item in ai_content:
                            if isinstance(item, dict):
                                if item.get('type') == 'tool_use':
                                    tool_name = item.get('name', 'unknown_tool')
                                    tool_input = item.get('input', {})
                                    tool_str = f"[TOOL_CALL: {tool_name}({json.dumps(tool_input, sort_keys=True)})]"
                                    content_parts.append(tool_str)
                                else:
                                    content_parts.append(json.dumps(item, sort_keys=True))
                            else:
                                content_parts.append(str(item))
                        content = " ".join(content_parts)
                    else:
                        content = str(ai_content)
                elif msg_type in ['tool', 'tool_response', 'observation']:
                    tool_content = msg.get('content', '')
                    if isinstance(tool_content, list) and len(tool_content) > 0:
                        if isinstance(tool_content[0], dict):
                            content = json.dumps(tool_content[0], sort_keys=True)
                        else:
                            content = str(tool_content[0])
                    else:
                        content = str(tool_content)

                    if 'name' in msg:
                        attributes['tool_name'] = msg['name']
                    elif 'id' in msg:
                         attributes['tool_call_id'] = msg['id']

                nodes.append(UniversalNode(content=content, node_type=node_type, attributes=attributes))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

        # Create sequential edges
        for i in range(1, len(nodes)):
             edges.append(UniversalEdge(source=i-1, target=i, edge_type='next_step'))

        # Infer specific edge types
        for i in range(1, len(nodes)):
            try:
                prev_type = nodes[i-1].node_type
                curr_type = nodes[i].node_type

                if prev_type == 'user' and curr_type == 'agent':
                    edges[-1].edge_type = 'handoff'
                elif prev_type == 'agent' and curr_type == 'tool':
                     edges[-1].edge_type = 'tool_call_observation'
                elif 'error' in nodes[i].content.lower():
                     edges[-1].edge_type = 'error_signal'
            except Exception as e:
                 logger.debug(f"Error inferring edge type: {e}")

        # Tool Call Linking (Heuristic)
        agent_indices = [i for i, n in enumerate(nodes) if n.node_type == 'agent']
        tool_indices = [i for i, n in enumerate(nodes) if n.node_type == 'tool']

        agent_idx_iter = iter(agent_indices)
        current_agent_idx = next(agent_idx_iter, None)

        for tool_idx in tool_indices:
            if current_agent_idx is not None and current_agent_idx < tool_idx:
                edges.append(UniversalEdge(source=current_agent_idx, target=tool_idx, edge_type='tool_call_observation'))
                current_agent_idx = next(agent_idx_iter, None)

        return nodes, edges

    def build_graph(self, embed_model: SentenceTransformer) -> nx.DiGraph:
        """
        Converts the UniversalTraceSchema into a NetworkX directed graph with features.
        """
        G = nx.DiGraph()
        if not self.nodes:
            logger.warning("No nodes to build graph from.")
            return G

        try:
            # --- 1. Generate Node Embeddings ---
            contents = [n.content for n in self.nodes]
            if contents:
                try:
                    embeddings = embed_model.encode(contents)
                except Exception as e:
                    logger.error(f"Error encoding content: {e}")
                    emb_dim = embed_model.get_sentence_embedding_dimension()
                    embeddings = np.zeros((len(contents), emb_dim))
            else:
                emb_dim = embed_model.get_sentence_embedding_dimension()
                embeddings = np.zeros((0, emb_dim))

            # --- 2. Create Node Type Vocabulary ---
            unique_node_types = list(set(n.node_type for n in self.nodes))
            type_to_id = {nt: i for i, nt in enumerate(unique_node_types)} if unique_node_types else {}

            # --- 3. Add Nodes with Features ---
            for i, node in enumerate(self.nodes):
                try:
                    role_id = float(type_to_id.get(node.node_type, len(type_to_id)))
                    if i < len(embeddings):
                        features = np.concatenate([
                            embeddings[i].astype(np.float32),
                            np.array([role_id, float(i)], dtype=np.float32)
                        ])
                    else:
                        emb_dim = embeddings.shape[1] if embeddings.size > 0 else embed_model.get_sentence_embedding_dimension()
                        features = np.array([0.0] * emb_dim + [role_id, float(i)], dtype=np.float32)
                    G.add_node(i, features=features, node_type=node.node_type)
                except Exception as e:
                    logger.error(f"Error adding node {i} to graph: {e}")
                    continue

            # --- 4. Create Edge Type Vocabulary ---
            unique_edge_types = list(set(e.edge_type for e in self.edges))
            edge_type_to_id = {et: i for i, et in enumerate(unique_edge_types)} if unique_edge_types else {}

            # --- 5. Add Edges with Attributes ---
            for e in self.edges:
                try:
                    if 0 <= e.source < len(G.nodes()) and 0 <= e.target < len(G.nodes()):
                        edge_type_id = edge_type_to_id.get(e.edge_type, len(edge_type_to_id))
                        G.add_edge(e.source, e.target, edge_type=e.edge_type, edge_type_id=edge_type_id)
                    else:
                       logger.warning(f"Edge index out of bounds: {e.source} -> {e.target}")
                except Exception as e:
                    logger.error(f"Error adding edge {e.source}->{e.target} to graph: {e}")
                    continue

        except Exception as e:
            logger.error(f"Unexpected error building graph: {e}", exc_info=True)

        return G

    def validate(self) -> bool:
        """Performs basic internal consistency checks."""
        try:
            for e in self.edges:
                if e.source < 0 or e.target < 0:
                    logger.warning(f"Edge has negative index: {e}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            return False

# --- Example Usage ---
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    # Example 1: internlm_agent format
    internlm_data = {
        "user_question": "What is 2+2?",
        "final_answer": "2+2 equals 4.",
        "steps": [
            {"type": "human", "content": "What is 2+2?"},
            {"type": "ai", "content": "I need to calculate this."},
            {"type": "tool", "content": "Result: 4", "name": "calculator"}
        ],
        "tools_used": ["calculator"],
        "errors": []
    }

    # Example 2: snorkelai_finance format
    snorkelai_data = {
        "initial_input": "Analyze this stock.",
        "final_output": "Here is the analysis.",
        "messages": [
            {"type": "user", "content": "Analyze this stock."},
            {"type": "assistant", "content": [{"type": "tool_use", "name": "get_stock_data", "input": {"symbol": "AAPL"}}]},
            {"type": "tool", "content": [{"price": 150.0}], "name": "get_stock_data"}
        ],
        "resources_used": ["get_stock_data"],
        "errors": None
    }

    # Example 3: WebShop-like format (steps + agent_actions list)
    webshop_data = {
        "user_question": "WebShop [SEP] Instruction: [SEP] i need a long clip-in hair extension...",
        "final_answer": "Thought: ... Action: click[buy now]",
        "steps": [
            {"type": "human", "content": "WebShop [SEP] Instruction: ..."},
            {"type": "gpt", "content": "OK"},
            {"type": "gpt", "content": "Thought: ... Action: click[...]"}, # Combined thought/action
        ],
        "agent_actions": [ # Explicit list of actions taken
            "Action: search[clip-in hair extension]",
            "Action: click[B07KD6QJ2G]",
            "Action: click[buy now]"
        ],
        "tools_used": ["click", "search"],
        "errors": []
    }


    print("--- Testing UniversalTraceSchema ---")
    print("\n1. Parsing internlm_agent format:")
    schema1 = UniversalTraceSchema.from_dict(internlm_data)
    if schema1:
        print(f"   Success: {len(schema1.nodes)} nodes, {len(schema1.edges)} edges")
        print("   Nodes:")
        for i, n in enumerate(schema1.nodes):
            print(f"     {i}: [{n.node_type}] '{n.content[:30]}...'")
    else:
        print("   Failed to parse.")

    print("\n2. Parsing snorkelai_finance format:")
    schema2 = UniversalTraceSchema.from_dict(snorkelai_data)
    if schema2:
        print(f"   Success: {len(schema2.nodes)} nodes, {len(schema2.edges)} edges")
        print("   Nodes:")
        for i, n in enumerate(schema2.nodes):
            print(f"     {i}: [{n.node_type}] '{n.content[:30]}...'")
    else:
        print("   Failed to parse.")

    print("\n3. Parsing WebShop-like format:")
    schema3 = UniversalTraceSchema.from_dict(webshop_data)
    if schema3:
        print(f"   Success: {len(schema3.nodes)} nodes, {len(schema3.edges)} edges")
        print("   Nodes:")
        for i, n in enumerate(schema3.nodes):
            print(f"     {i}: [{n.node_type}] '{n.content[:50]}...' {n.attributes}")
        print("   Edges:")
        for e in schema3.edges:
             print(f"     {e.source} -> {e.target} [{e.edge_type}]")
        print(f"   Resources Used: {schema3.resources_used}")
    else:
        print("   Failed to parse.")

    print("\n--- Testing Graph Building ---")
    try:
        model = SentenceTransformer(TRANSFORMER_NAME)
        if schema1:
            g1 = schema1.build_graph(model)
            print(f"\nGraph 1 built: {len(g1.nodes())} nodes, {len(g1.edges())} edges")
        if schema2:
            g2 = schema2.build_graph(model)
            print(f"Graph 2 built: {len(g2.nodes())} nodes, {len(g2.edges())} edges")
        if schema3:
            g3 = schema3.build_graph(model)
            print(f"Graph 3 (WebShop) built: {len(g3.nodes())} nodes, {len(g3.edges())} edges")
    except Exception as e:
        print(f"Error building graphs: {e}")
