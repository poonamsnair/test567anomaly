import os
import json
import random
import argparse
import numpy as np
from datasets import load_dataset  # pip install datasets

NORMAL_DIR = 'training_dataset/normal/'
os.makedirs(NORMAL_DIR, exist_ok=True)

ANOMALY_TYPES = [
    "Planning Failure", "Decomposition Error", "Tool Calling Error", "Inadequate Validation of Tool Results",
    "Suboptimal Path", "Agent Handoff Error", "Feedback Loop Failure", "Memory Inconsistency",
    "Error Propagation", "Task Not Completed", "Partial or Incomplete Answer", "Irrelevant or Off-Topic Answer",
    "Overconfidence in Incorrect Answer", "Loop or Repetition", "Misinterpretation of Question or Context",
    "Lack of Alternative Strategy", "Failure to Recover from Error", "Hallucination",
    "Handling of Fictional or Impossible Queries"
]

def load_agentbank_data(dataset_name='Solaris99/AgentBank', config='gsm8k', split='train'):
    try:
        ds = load_dataset(dataset_name, config, split=split)
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'datasets' library is installed and dataset is available.")
        exit(1)

def modify_to_schema(trajectory, is_anomaly=False):
    trace = {}
    conversations = trajectory.get('conversations', [])
    if not conversations:
        return None
    # First step: user question (first 'human' message)
    first_human = next((c for c in conversations if c['from'] == 'human'), None)
    if not first_human:
        return None
    steps = [{
        "step_number": 1,
        "type": "task",
        "role": "user",
        "agent": "user",
        "content": first_human['value']
    }]
    # Remaining steps: skip the first human, process the rest
    step_num = 2
    for c in conversations[1:]:
        role = "user" if c['from'] == 'human' else "agent"
        agent = "user" if c['from'] == 'human' else "gpt"
        value = c['value']
        # Infer type and tool
        if value.lower().startswith('thought:'):
            step_type = "analysis"
            tool = None
        elif value.lower().startswith('action:'):
            step_type = "tool_call"
            tool = value.split(':', 1)[-1].strip().split()[0]
        elif value.lower().startswith('observation:'):
            step_type = "observation"
            tool = None
        elif value.lower().startswith('final answer:'):
            step_type = "answer"
            tool = None
        else:
            step_type = "observation"
            tool = None
        steps.append({
            "step_number": step_num,
            "type": step_type,
            "role": role,
            "agent": agent,
            "tool": tool,
            "content": value
        })
        step_num += 1
    # Metadata synthesis
    num_steps = len(steps)
    agents_called = list({s['agent'] for s in steps})
    tools_called = list({s['tool'] for s in steps if s.get('tool')})
    duration = random.randint(5, 30)
    errors = [] if not is_anomaly else [f"Simulated {random.choice(ANOMALY_TYPES)}"]
    task_completed = not is_anomaly
    anomaly_types = [random.choice(ANOMALY_TYPES)] if is_anomaly else []
    # Memory state: Synthetic plan and knowledge
    plan_steps = []
    for j in range(1, num_steps + 1):
        completed = random.choice([True, False]) if is_anomaly else True
        plan_steps.append({
            "id": j,
            "description": f"Step {j}: Perform action {j}",
            "completed": completed
        })
    shared_knowledge = "Aggregated insights from trajectory observations."
    metadata = {
        "agents_called": agents_called,
        "duration": f"{duration} minutes",
        "errors": errors,
        "num_steps": num_steps,
        "task_completed": task_completed,
        "tools_called": tools_called,
        "anomaly_types": anomaly_types,
        "memory_state": {
            "plan": {
                "steps": plan_steps,
                "current_objective": "Complete the assigned task"
            },
            "shared_knowledge": shared_knowledge
        }
    }
    trace['question'] = first_human['value']
    trace['metadata'] = metadata
    trace['steps'] = steps
    trace['trace_id'] = f"adapted_agentbank_{'anomaly' if is_anomaly else 'normal'}_{random.randint(10000, 99999)}"
    # Perturb for anomalies
    if is_anomaly:
        if random.random() > 0.5:
            duplicate_step = random.choice(steps)
            steps.append(duplicate_step.copy())
            metadata['num_steps'] += 1
        random_plan = random.choice(metadata['memory_state']['plan']['steps'])
        random_plan['completed'] = not random_plan['completed']
    return trace

def generate_and_save(dataset, num_samples=5):
    # Find the next available trace index for normal
    existing_normal = [f for f in os.listdir(NORMAL_DIR) if f.startswith('agentbank_normal_') and f.endswith('.json')]
    if existing_normal:
        indices_normal = [int(f.split('_')[-1].split('.')[0]) for f in existing_normal if f.split('_')[-1].split('.')[0].isdigit()]
        start_idx_normal = max(indices_normal) + 1 if indices_normal else 0
    else:
        start_idx_normal = 0
    # Find the next available trace index for anomaly
    ANOMALY_DIR = 'training_dataset/anomaly/'
    os.makedirs(ANOMALY_DIR, exist_ok=True)
    existing_anomaly = [f for f in os.listdir(ANOMALY_DIR) if f.startswith('agentbank_anomaly_') and f.endswith('.json')]
    if existing_anomaly:
        indices_anomaly = [int(f.split('_')[-1].split('.')[0]) for f in existing_anomaly if f.split('_')[-1].split('.')[0].isdigit()]
        start_idx_anomaly = max(indices_anomaly) + 1 if indices_anomaly else 0
    else:
        start_idx_anomaly = 0
    sampled = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    normal_count = start_idx_normal
    anomaly_count = start_idx_anomaly
    for idx in sampled:
        traj = dataset[idx]
        # Check for error in any conversation value
        has_error = any('error' in c['value'].lower() for c in traj.get('conversations', []))
        if has_error:
            modified_trace = modify_to_schema(traj, is_anomaly=True)
            if modified_trace:
                trace_id = f"agentbank_anomaly_{anomaly_count:04d}"
                out_path = os.path.join(ANOMALY_DIR, f"{trace_id}.json")
                modified_trace['trace_id'] = trace_id
                with open(out_path, 'w') as f:
                    json.dump(modified_trace, f, indent=2)
                print(f"Generated and saved anomaly: {out_path}")
                anomaly_count += 1
            else:
                print(f"Skipped trajectory {idx} due to missing conversations.")
        else:
            modified_trace = modify_to_schema(traj, is_anomaly=False)
            if modified_trace:
                trace_id = f"agentbank_normal_{normal_count:04d}"
                out_path = os.path.join(NORMAL_DIR, f"{trace_id}.json")
                modified_trace['trace_id'] = trace_id
                with open(out_path, 'w') as f:
                    json.dump(modified_trace, f, indent=2)
                print(f"Generated and saved: {out_path}")
                normal_count += 1
            else:
                print(f"Skipped trajectory {idx} due to missing conversations.")

def save_specific_anomalies(dataset, indices):
    ANOMALY_DIR = 'training_dataset/anomaly/'
    os.makedirs(ANOMALY_DIR, exist_ok=True)
    existing_anomaly = [f for f in os.listdir(ANOMALY_DIR) if f.startswith('agentbank_anomaly_') and f.endswith('.json')]
    if existing_anomaly:
        indices_anomaly = [int(f.split('_')[-1].split('.')[0]) for f in existing_anomaly if f.split('_')[-1].split('.')[0].isdigit()]
        start_idx_anomaly = max(indices_anomaly) + 1 if indices_anomaly else 0
    else:
        start_idx_anomaly = 0
    anomaly_count = start_idx_anomaly
    for idx in indices:
        traj = dataset[idx]
        modified_trace = modify_to_schema(traj, is_anomaly=True)
        if modified_trace:
            trace_id = f"agentbank_anomaly_{anomaly_count:04d}"
            out_path = os.path.join(ANOMALY_DIR, f"{trace_id}.json")
            modified_trace['trace_id'] = trace_id
            with open(out_path, 'w') as f:
                json.dump(modified_trace, f, indent=2)
            print(f"Generated and saved anomaly: {out_path}")
            anomaly_count += 1
        else:
            print(f"Skipped trajectory {idx} due to missing conversations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGENTBANK to Normal Schema Converter")
    parser.add_argument('--num_samples', type=int, default=5, help='Number of normal samples to generate')
    parser.add_argument('--config', type=str, default='gsm8k', help='AgentBank config to use (e.g., gsm8k, hotpotqa)')
    parser.add_argument('--only_anomaly', action='store_true', help='Only save known error anomalies')
    args = parser.parse_args()
    dataset = load_agentbank_data(config=args.config, split='train')
    if args.only_anomaly:
        # Hardcoded indices for gsm8k with errors
        save_specific_anomalies(dataset, [2480, 5563])
    else:
        generate_and_save(dataset, args.num_samples) 