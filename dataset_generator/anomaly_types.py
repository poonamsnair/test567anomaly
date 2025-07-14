"""
Standardized Anomaly Types for Multi-Agent Trajectory Analysis

This module defines the complete list of anomaly types that can occur in multi-agent
trajectories, organized by thematic categories. Each anomaly type includes a description,
step-level characteristics, and detection methods.

The script also includes LLM functionality to automatically update existing anomaly
data files to use the standardized anomaly types.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to import openai, but handle case where it's not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. LLM functionality will not be available.")

# Standardized Anomaly Types List
ANOMALY_TYPES = [
    # Planning and Decomposition Anomalies
    "Planning Failure",
    "Decomposition Error",
    
    # Execution and Tool-Related Anomalies
    "Tool Calling Error",
    "Inadequate Validation of Tool Results",
    "Suboptimal Path",
    
    # Collaboration and Handoff Anomalies
    "Agent Handoff Error",
    "Feedback Loop Failure",
    
    # Memory and State Anomalies
    "Memory Inconsistency",
    "Error Propagation",
    
    # Task Completion and Output Anomalies
    "Task Not Completed",
    "Partial or Incomplete Answer",
    "Irrelevant or Off-Topic Answer",
    "Overconfidence in Incorrect Answer",
    
    # Behavioral and Reasoning Anomalies
    "Loop or Repetition",
    "Misinterpretation of Question or Context",
    "Lack of Alternative Strategy",
    "Failure to Recover from Error",
    
    # Specialized Anomalies
    "Hallucination",
    "Handling of Fictional or Impossible Queries"
]

# Detailed descriptions for each anomaly type
ANOMALY_DESCRIPTIONS = {
    # Planning and Decomposition Anomalies
    "Planning Failure": "Errors in creating, updating, or managing plans, such as incomplete objective shifts or deleted steps without rationale. Step-level: Occurs in plan-related steps (e.g., 'create_plan' tool misuse). Detection: High tool_entropy or reconstruction error in plan_update steps.",
    
    "Decomposition Error": "Faulty breakdown of tasks into sub-tasks, leading to mismatched or redundant assignments. Step-level: Manifests in decomposition/assignment steps. Detection: Unusual motif patterns (e.g., unbalanced subgraphs) or agent_switching_frequency spikes.",
    
    # Execution and Tool-Related Anomalies
    "Tool Calling Error": "Incorrect or inappropriate tool invocation, like using 'search' for computation. Step-level: Isolated to tool_call steps. Detection: Keyword features (e.g., contains_failed) or inadequate_validation in observations.",
    
    "Inadequate Validation of Tool Results": "Failure to verify or cross-check tool outputs, accepting flawed data. Step-level: Follows tool_call/observation pairs. Detection: Semantic embedding_std deviations or error_propagation in subsequent steps.",
    
    "Suboptimal Path": "Inefficient sequencing of steps or tools, causing roundabout execution without failure. Step-level: Across multiple steps (e.g., unnecessary repeats). Detection: High steps_per_minute or tool_usage_frequency anomalies.",
    
    # Collaboration and Handoff Anomalies
    "Agent Handoff Error": "Incorrect assignment or handoff to the wrong sub-agent (e.g., researcher handling code). Step-level: In assignment/feedback steps. Detection: Agent_entropy spikes or max_agent_streak deviations.",
    
    "Feedback Loop Failure": "Broken or ineffective feedback between agents, ignoring corrections. Step-level: In feedback/plan_update cycles. Detection: Loop detection in motifs or consecutive_repeat_count.",
    
    # Memory and State Anomalies
    "Memory Inconsistency": "Inconsistent recall/updating of shared memory/state (e.g., outdated plan status). Step-level: In memory_update or observation steps. Detection: GMM OOD on latents (novel state mismatches) or content_embedding_std.",
    
    "Error Propagation": "Errors from one step carry over without mitigation, compounding issues. Step-level: Chains across steps (e.g., bad data reused). Detection: Cumulative recon errors in score timelines.",
    
    # Task Completion and Output Anomalies
    "Task Not Completed": "Abrupt end without final_answer or resolution. Step-level: Missing/incomplete final steps. Detection: Low task_completed flag + high num_steps variance.",
    
    "Partial or Incomplete Answer": "Output is fragmented, missing key elements despite 'completion.' Step-level: In answer/final_answer steps. Detection: Semantic mean/std mismatches or question_tool_match false.",
    
    "Irrelevant or Off-Topic Answer": "Response deviates from query (e.g., hallucinated tangents). Step-level: In analysis/answer steps. Detection: High cosine_error in embeddings.",
    
    "Overconfidence in Incorrect Answer": "Asserting wrong info as correct without validation. Step-level: In observation/answer steps. Detection: Keyword flags (e.g., contains_exception) + recon error.",
    
    # Behavioral and Reasoning Anomalies
    "Loop or Repetition": "Repeated actions/steps without progress (e.g., endless searches). Step-level: Consecutive similar steps. Detection: Motif repeats or repetitive_actions count.",
    
    "Misinterpretation of Question or Context": "Wrong understanding of input, leading to misdirected steps. Step-level: Early in interpret/decompose. Detection: Question_complexity vs. step_complexity mismatch.",
    
    "Lack of Alternative Strategy": "No fallback on failure, causing dead-ends. Step-level: After error/observation. Detection: Failure_to_recover via error propagation.",
    
    "Failure to Recover from Error": "Ignores/dismisses errors without retry. Step-level: Post-error steps. Detection: Contains_error + no plan_update.",
    
    # Specialized Anomalies
    "Hallucination": "Generating unsupported info (e.g., fabricated facts without tools). Step-level: In reasoning/answer steps. Detection: Semantic deviations (embed_std) or no tool_validation.",
    
    "Handling of Fictional or Impossible Queries": "Poor response to nonsensical inputs (e.g., loops on invalid queries). Step-level: Full trace, but starts early. Detection: High OOD from GMM (novel content)."
}

# Thematic groupings for organization
ANOMALY_CATEGORIES = {
    "Planning and Decomposition": [
        "Planning Failure",
        "Decomposition Error"
    ],
    "Execution and Tool-Related": [
        "Tool Calling Error",
        "Inadequate Validation of Tool Results",
        "Suboptimal Path"
    ],
    "Collaboration and Handoff": [
        "Agent Handoff Error",
        "Feedback Loop Failure"
    ],
    "Memory and State": [
        "Memory Inconsistency",
        "Error Propagation"
    ],
    "Task Completion and Output": [
        "Task Not Completed",
        "Partial or Incomplete Answer",
        "Irrelevant or Off-Topic Answer",
        "Overconfidence in Incorrect Answer"
    ],
    "Behavioral and Reasoning": [
        "Loop or Repetition",
        "Misinterpretation of Question or Context",
        "Lack of Alternative Strategy",
        "Failure to Recover from Error"
    ],
    "Specialized": [
        "Hallucination",
        "Handling of Fictional or Impossible Queries"
    ]
}

class AnomalyTypeUpdater:
    """LLM-powered updater for standardizing anomaly types in existing datasets."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the anomaly type updater.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: LLM model to use for classification.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Please install it with: pip install openai")
            
        self.model = model
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            env_api_key = os.getenv('OPENAI_API_KEY')
            if not env_api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            self.api_key = env_api_key
    
    def _create_classification_prompt(self, trace_data: Dict[str, Any]) -> str:
        """Create a prompt for the LLM to classify anomaly types."""
        
        # Extract key information from the trace
        question = trace_data.get('question', '')
        steps = trace_data.get('steps', [])
        metadata = trace_data.get('metadata', {})
        
        # Create a summary of the trace
        step_summary = []
        for i, step in enumerate(steps[:10]):  # Limit to first 10 steps for brevity
            step_summary.append(f"Step {i+1}: {step.get('type', 'unknown')} - {step.get('content', '')[:100]}...")
        
        if len(steps) > 10:
            step_summary.append(f"... and {len(steps) - 10} more steps")
        
        prompt = f"""
You are an expert in analyzing multi-agent trajectories and identifying anomaly types.

Given the following trace data, please identify which anomaly type (from the standardized list) is the single best match for the anomaly in this trace. Only if the trace clearly and unambiguously exhibits multiple distinct anomaly types, you may add a second or (rarely) a third. In most cases, return only one anomaly type.

TRACE DATA:
Question: {question}
Task Completed: {metadata.get('task_completed', 'Unknown')}
Number of Steps: {metadata.get('num_steps', 'Unknown')}
Errors: {metadata.get('errors', [])}

Steps Summary:
{chr(10).join(step_summary)}

STANDARDIZED ANOMALY TYPES:
{chr(10).join([f"- {anomaly_type}: {ANOMALY_DESCRIPTIONS[anomaly_type].split('Step-level:')[0].strip()}" for anomaly_type in ANOMALY_TYPES])}

INSTRUCTIONS:
1. Analyze the trace for anomalous behavior
2. Select the single best anomaly type from the standardized list
3. Only add a second or third type if the trace clearly and unambiguously exhibits multiple distinct anomaly types
4. Return ONLY a JSON array of the selected anomaly type names (e.g., ["Tool Calling Error"])
5. If no clear anomalies are present, return ["Normal"]

Example response: ["Tool Calling Error"]

Your classification:
"""
        return prompt
    
    def classify_trace(self, trace_data: Dict[str, Any]) -> List[str]:
        """
        Use LLM to classify anomaly types for a given trace.
        
        Args:
            trace_data: The trace data dictionary
            
        Returns:
            List of anomaly type names
        """
        try:
            prompt = self._create_classification_prompt(trace_data)
            
            # Use the new OpenAI API format
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in multi-agent trajectory analysis and anomaly detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Extract the response
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                import ast
                # Handle both JSON and Python list formats
                if content.startswith('[') and content.endswith(']'):
                    # Try to evaluate as Python literal
                    result = ast.literal_eval(content)
                    if isinstance(result, list):
                        # Enforce: prefer 1, only allow >1 if LLM is very confident
                        if len(result) > 1:
                            # If the LLM output is just a list, keep only the first
                            return [result[0]]
                        return result
                # Try JSON parsing
                import json
                result = json.loads(content)
                if isinstance(result, list):
                    if len(result) > 1:
                        return [result[0]]
                    return result
            except:
                pass
            
            # Fallback: try to extract anomaly types from text
            detected_types = []
            for anomaly_type in ANOMALY_TYPES:
                if anomaly_type.lower() in content.lower():
                    detected_types.append(anomaly_type)
            if detected_types:
                return [detected_types[0]]
            return ["Normal"]
            
        except Exception as e:
            print(f"Error classifying trace: {e}")
            return ["Normal"]
    
    def update_trace_file(self, file_path: str, backup: bool = False) -> bool:
        """
        Update a single trace file with standardized anomaly types.
        
        Args:
            file_path: Path to the trace file
            backup: Whether to create a backup before updating (default: False)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the trace file
            with open(file_path, 'r') as f:
                trace_data = json.load(f)
            
            # Create backup if requested
            if backup:
                backup_path = file_path + '.backup'
                with open(backup_path, 'w') as f:
                    json.dump(trace_data, f, indent=2)
            
            # Classify the trace
            anomaly_types = self.classify_trace(trace_data)
            
            # Update the metadata
            if 'metadata' not in trace_data:
                trace_data['metadata'] = {}
            
            trace_data['metadata']['anomaly_types'] = anomaly_types
            
            # Write back the updated file
            with open(file_path, 'w') as f:
                json.dump(trace_data, f, indent=2)
            
            print(f"Updated {file_path}: {anomaly_types}")
            return True
            
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            return False
    
    def update_dataset_directory(self, directory_path: str, pattern: str = "*.json", backup: bool = False) -> Dict[str, Any]:
        """
        Update all trace files in a directory with standardized anomaly types.
        
        Args:
            directory_path: Path to the directory containing trace files
            pattern: File pattern to match (default: "*.json")
            backup: Whether to create backups before updating
            
        Returns:
            Dictionary with update statistics
        """
        stats = {
            'total_files': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'updated_types': {}
        }
        
        # Find all matching files
        file_pattern = os.path.join(directory_path, pattern)
        files = glob.glob(file_pattern)
        
        print(f"Found {len(files)} files to update in {directory_path}")
        
        for file_path in files:
            stats['total_files'] += 1
            
            if self.update_trace_file(file_path, backup):
                stats['successful_updates'] += 1
                
                # Read the updated file to get the new anomaly types
                try:
                    with open(file_path, 'r') as f:
                        updated_data = json.load(f)
                    anomaly_types = updated_data.get('metadata', {}).get('anomaly_types', [])
                    for anomaly_type in anomaly_types:
                        stats['updated_types'][anomaly_type] = stats['updated_types'].get(anomaly_type, 0) + 1
                except:
                    pass
            else:
                stats['failed_updates'] += 1
        
        return stats

def get_anomaly_types():
    """Return the complete list of standardized anomaly types."""
    return ANOMALY_TYPES.copy()

def get_anomaly_description(anomaly_type):
    """Get the description for a specific anomaly type."""
    return ANOMALY_DESCRIPTIONS.get(anomaly_type, "Description not available")

def get_anomaly_categories():
    """Return the thematic groupings of anomaly types."""
    return ANOMALY_CATEGORIES.copy()

def validate_anomaly_type(anomaly_type):
    """Validate if an anomaly type is in the standardized list."""
    return anomaly_type in ANOMALY_TYPES

def get_anomaly_types_by_category(category):
    """Get anomaly types for a specific category."""
    return ANOMALY_CATEGORIES.get(category, [])

def print_anomaly_types_summary():
    """Print a formatted summary of all anomaly types."""
    print("Standardized Anomaly Types for Multi-Agent Trajectory Analysis")
    print("=" * 70)
    print()
    
    for category, types in ANOMALY_CATEGORIES.items():
        print(f"{category} Anomalies:")
        print("-" * 40)
        for anomaly_type in types:
            description = ANOMALY_DESCRIPTIONS[anomaly_type]
            # Extract the main description (before "Step-level:")
            main_desc = description.split("Step-level:")[0].strip()
            print(f"  • {anomaly_type}: {main_desc}")
        print()

def update_all_datasets(api_key: Optional[str] = None, backup: bool = False):
    """
    Update all anomaly datasets with standardized anomaly types.
    
    Args:
        api_key: OpenAI API key
        backup: Whether to create backups before updating
    """
    updater = AnomalyTypeUpdater(api_key)
    
    # Define the directories to update
    directories = [
        "../training_dataset/anomaly/",
        "../eval_dataset/anomaly/"
    ]
    
    total_stats = {
        'total_files': 0,
        'successful_updates': 0,
        'failed_updates': 0,
        'updated_types': {}
    }
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nUpdating {directory}...")
            stats = updater.update_dataset_directory(directory, backup=backup)
            
            # Aggregate statistics
            total_stats['total_files'] += stats['total_files']
            total_stats['successful_updates'] += stats['successful_updates']
            total_stats['failed_updates'] += stats['failed_updates']
            
            for anomaly_type, count in stats['updated_types'].items():
                total_stats['updated_types'][anomaly_type] = total_stats['updated_types'].get(anomaly_type, 0) + count
            
            print(f"Directory {directory} completed:")
            print(f"  - Files processed: {stats['total_files']}")
            print(f"  - Successful updates: {stats['successful_updates']}")
            print(f"  - Failed updates: {stats['failed_updates']}")
        else:
            print(f"Directory {directory} not found, skipping...")
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {total_stats['total_files']}")
    print(f"Successful updates: {total_stats['successful_updates']}")
    print(f"Failed updates: {total_stats['failed_updates']}")
    print(f"Success rate: {(total_stats['successful_updates']/total_stats['total_files']*100):.1f}%")
    
    print(f"\nAnomaly type distribution:")
    for anomaly_type, count in sorted(total_stats['updated_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {anomaly_type}: {count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update anomaly datasets with standardized anomaly types")
    parser.add_argument("--action", choices=["summary", "update"], default="summary",
                       help="Action to perform: summary (print types) or update (update datasets)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    parser.add_argument("--directory", type=str, help="Specific directory to update (optional)")
    
    args = parser.parse_args()
    
    if args.action == "summary":
        print_anomaly_types_summary()
    elif args.action == "update":
        if args.directory:
            # Update specific directory
            updater = AnomalyTypeUpdater(args.api_key)
            if os.path.exists(args.directory):
                stats = updater.update_dataset_directory(args.directory, backup=False)
                print(f"Updated {args.directory}: {stats}")
            else:
                print(f"Directory {args.directory} not found")
        else:
            # Update all datasets
            update_all_datasets(args.api_key, backup=False) 