import os
import json
from openai import OpenAI
import re
import argparse

# Check for OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    print("ERROR: OpenAI API key not found!")
    exit(1)

client = OpenAI()

NORMAL_DIR = 'training_dataset/normal/'
ANOMALY_DIR = 'training_dataset/anomaly/'
MODEL = 'gpt-4o'
NUM_NORMAL = 50
NUM_ANOMALY_PER_TYPE = 3  # Adjust as needed; total anomalies = this * num_types, aim for ~50/20 ≈ 3

AVAILABLE_TOOLS = [
    "search", "code", "image_generation", "code_generation", "final_answer",
    "create_plan", "update_plan", "delete_plan", "mark_step_completed",
    "interpret", "decompose", "assign", "feedback"
]
AVAILABLE_AGENTS = [
    "user", "principal_agent", "search_agent", "code_agent", "media_agent",
    "planner", "researcher", "browser", "analyzer",
    "deep_researcher_agent", "browser_use_agent", "deep_analyzer_agent", "other_sub_agent"
]

# Import standardized anomaly types
from anomaly_types import ANOMALY_TYPES, ANOMALY_DESCRIPTIONS

# Use imported descriptions from anomaly_types

FEWSHOT_NORMAL = '''
{
  "metadata": {
    "agents_called": [
      "user",
      "planner",
      "researcher",
      "browser",
      "analyzer"
    ],
    "duration": "12 minutes",
    "errors": [],
    "num_steps": 15,
    "task_completed": true,
    "tools_called": [
      "create_plan",
      "decompose",
      "assign",
      "search",
      "feedback",
      "update_plan",
      "final_answer"
    ],
    "memory_state": {
      "plan": {
        "steps": [
          {"id": 1, "description": "Research topic", "completed": true},
          {"id": 2, "description": "Analyze findings", "completed": true},
          {"id": 3, "description": "Generate report", "completed": true}
        ],
        "current_objective": "Complete analysis of underwater basket weaving"
      },
      "shared_knowledge": "Underwater basket weaving originated in coastal communities; certifications include Aquatic Artistry."
    }
  },
  "question": "Provide a detailed report on the history and certifications of underwater basket weaving.",
  "steps": [
    {
      "agent": "user",
      "content": "Provide a detailed report on the history and certifications of underwater basket weaving.",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "planner",
      "content": "Creating initial plan: 1) Interpret query. 2) Decompose into research and analysis. 3) Assign to sub-agents.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_plan",
      "type": "plan"
    },
    {
      "agent": "planner",
      "content": "Decomposing task into sub-tasks: history research, certification lookup, synthesis.",
      "role": "agent",
      "step_number": 3,
      "tool": "decompose",
      "type": "decomposition"
    },
    {
      "agent": "planner",
      "content": "Assigning history research to researcher agent.",
      "role": "agent",
      "step_number": 4,
      "tool": "assign",
      "type": "assignment"
    },
    {
      "agent": "researcher",
      "content": "Searching for history of underwater basket weaving.",
      "role": "agent",
      "step_number": 5,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "researcher",
      "content": "History: Originated as artistic expression in coastal communities.",
      "role": "agent",
      "step_number": 6,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Updating plan: Mark history research as completed.",
      "role": "agent",
      "step_number": 7,
      "tool": "mark_step_completed",
      "type": "plan_update"
    },
    {
      "agent": "browser",
      "content": "Browsing for certifications.",
      "role": "agent",
      "step_number": 8,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "browser",
      "content": "Certifications: Aquatic Artistry Certification, Submerged Craftsmanship Diploma.",
      "role": "agent",
      "step_number": 9,
      "type": "observation"
    },
    {
      "agent": "analyzer",
      "content": "Analyzing combined findings.",
      "role": "agent",
      "step_number": 10,
      "type": "analysis"
    },
    {
      "agent": "analyzer",
      "content": "Providing feedback to planner: Analysis complete, ready for report.",
      "role": "agent",
      "step_number": 11,
      "tool": "feedback",
      "type": "feedback"
    },
    {
      "agent": "planner",
      "content": "Updating plan based on feedback.",
      "role": "agent",
      "step_number": 12,
      "tool": "update_plan",
      "type": "plan_update"
    },
    {
      "agent": "principal_agent",
      "content": "Compiling final report: History and certifications detailed.",
      "role": "agent",
      "step_number": 13,
      "tool": "final_answer",
      "type": "answer"
    },
    {
      "agent": "planner",
      "content": "Task completed successfully.",
      "role": "agent",
      "step_number": 14,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Updating memory with key insights.",
      "role": "agent",
      "step_number": 15,
      "type": "memory_update"
    }
  ],
  "trace_id": "trace_example_complex_003",
  "true_answer": "Detailed report on history and certifications of underwater basket weaving provided."
}
'''

FEWSHOT_ANOMALY = '''
{
  "metadata": {
    "agents_called": [
      "user",
      "planner",
      "researcher"
    ],
    "duration": "15 minutes",
    "errors": [
      "Failed to update plan correctly, leading to inconsistency."
    ],
    "num_steps": 10,
    "task_completed": false,
    "tools_called": [
      "create_plan",
      "decompose",
      "search"
    ],
    "anomaly_types": [
      "Planning Failure",
      "Memory Inconsistency"
    ],
    "memory_state": {
      "plan": {
        "steps": [
          {"id": 1, "description": "Research topic", "completed": false},
          {"id": 2, "description": "Outdated step", "completed": true}  // Inconsistent with updates
        ],
        "current_objective": "Incomplete due to memory mismatch"
      }
    }
  },
  "question": "Analyze the impact of Arctic ice melting on ocean salinity.",
  "steps": [
    {
      "agent": "user",
      "content": "Analyze the impact of Arctic ice melting on ocean salinity.",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "planner",
      "content": "Creating plan: 1) Decompose query. 2) Research effects.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_plan",
      "type": "plan"
    },
    {
      "agent": "planner",
      "content": "Decomposing into sub-tasks, but incorrectly including irrelevant current effects.",
      "role": "agent",
      "step_number": 3,
      "tool": "decompose",
      "type": "decomposition"
    },
    {
      "agent": "researcher",
      "content": "Searching for ice melting effects.",
      "role": "agent",
      "step_number": 4,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "researcher",
      "content": "Found info on currents, but not salinity.",
      "role": "agent",
      "step_number": 5,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Attempting to update plan, but forgetting previous decomposition.",
      "role": "agent",
      "step_number": 6,
      "tool": "update_plan",
      "type": "plan_update"
    },
    {
      "agent": "planner",
      "content": "Memory inconsistency: Plan state not reflecting updates.",
      "role": "agent",
      "step_number": 7,
      "type": "error"
    },
    {
      "agent": "researcher",
      "content": "Continuing with outdated plan, leading to irrelevant search.",
      "role": "agent",
      "step_number": 8,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "planner",
      "content": "Task failed due to planning and memory issues.",
      "role": "agent",
      "step_number": 9,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Failed to recover; inconsistent memory state.",
      "role": "agent",
      "step_number": 10,
      "type": "error"
    }
  ],
  "trace_id": "trace_example_anomaly_005",
  "true_answer": "The melting of Arctic ice decreases ocean salinity by adding freshwater."
}
'''

NORMAL_PROMPT_TEMPLATE = f'''
You are an expert at generating synthetic agent traces for a reasoning pipeline in a hierarchical role-based multi-agent system like AgentOrchestra. Generate 1 synthetic trace in the following JSON schema, where:
- The user asks a normal question, possibly with follow-ups, interruptions, or multi-turn conversations in real-life scenarios.
- Agents collaborate hierarchically: Planner decomposes tasks, assigns to sub-agents (e.g., researcher, browser, analyzer), uses feedback, and updates plans dynamically.
- Include a memory component: "memory_state" in metadata for persistent plan status, shared knowledge, or current objective. Update it via steps like memory_update or plan changes.
- Show realistic, successful flows: correct tool calls (e.g., create_plan, decompose, assign, search), agent handoffs, feedback loops, objective shifts, and error-free execution.
- Vary lengths: short single-turn or long multi-turn with user interactions, diverse topics (e.g., research, code, images, planning complex tasks).
- Task completed true, no errors, realistic to real-life use cases with dynamic planning and memory persistence.

IMPORTANT CONSTRAINTS:
- ONLY use these tools: {', '.join(AVAILABLE_TOOLS)}
- ONLY use these agents: {', '.join(AVAILABLE_AGENTS)}
- Interaction should be nuanced, collaborative, and realistic for training anomaly detection—include plan updates, feedback, and memory consistency.
- Agents plan, decompose, assign, observe, reason, hand off, provide feedback, update memory; users can interrupt or ask follow-ups.
- Output only the raw JSON object with no additional text.

Schema:
{{
  "trace_id": "...",
  "question": "...",
  "true_answer": "...",
  "metadata": {{
    "task_completed": true,
    "errors": [],
    "duration": "...",
    "tools_called": ["..."],
    "num_steps": ...,
    "agents_called": ["..."],
    "memory_state": {{ /* Persistent state: plan steps with completion status, shared_knowledge, current_objective */ }}
  }},
  "steps": [
    {{
      "step_number": ...,
      "type": "...",  // plan, decomposition, assignment, tool_call, observation, feedback, plan_update, memory_update, answer, question, task, etc.
      "role": "...",  // agent or user
      "agent": "...",
      "tool": "...",  // if applicable
      "content": "..."
    }}
  ]
}}

Here are some examples:
{FEWSHOT_NORMAL}

Generate a trace that follows the same structure and style as the examples, but with a different scenario, diverse length, and realistic interactions including memory updates.
'''

ANOMALY_PROMPT_TEMPLATE_BASE = '''
You are an expert at generating synthetic agent traces for a reasoning pipeline in a hierarchical role-based multi-agent system like AgentOrchestra. Generate 1 synthetic trace in the following JSON schema, where:
- The user asks a question leading to an anomalous trajectory with hierarchical issues.
- Agents exhibit anomalies: e.g., planning failures, wrong decompositions, feedback breaks, memory inconsistencies (e.g., outdated state recall).
- Include memory component: "memory_state" in metadata, but show inconsistencies or failures in updating it for anomalies.
- Trace nuanced and dynamic for adaptive anomaly scoring—task completed false, with errors and anomaly_types.
- Realistic to real-life: failed collaborations, objective shifts without updates, sub-agent errors propagating.

IMPORTANT CONSTRAINTS:
- ONLY use these tools: {tools}
- ONLY use these agents: {agents}
- Use realistic structure but inject anomalies; include hierarchical elements like planning, sub-agents, feedback.
- Agents plan/decompose/assign/observe/reason/hand off/feedback/update memory; users interrupt, but lead to anomaly.
- Output only the raw JSON object with no additional text.

Schema:
{{
  "trace_id": "...",
  "question": "...",
  "true_answer": "...",
  "metadata": {{
    "task_completed": false,
    "errors": ["..."],
    "duration": "...",
    "tools_called": ["..."],
    "num_steps": ...,
    "agents_called": ["..."],
    "anomaly_types": ["..."],
    "memory_state": {{ /* Potentially inconsistent state for anomalies */ }}
  }},
  "steps": [
    {{
      "step_number": ...,
      "type": "...",  // plan, decomposition, assignment, tool_call, observation, feedback, plan_update, memory_update, error, answer, question, task, etc.
      "role": "...",  // agent or user
      "agent": "...",
      "tool": "...",  // if applicable
      "content": "..."
    }}
  ]
}}

Here are some examples:
{examples}

Generate a trace that follows the same structure and style as the examples, but with a different scenario.
Make sure the trace demonstrates the following anomaly: {category}: {description}
Include the anomaly type in metadata["anomaly_types"], possibly with related types. Show memory inconsistencies where relevant.
'''

def get_next_id(directory, prefix):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.json')]
    if not files:
        return 1
    ids = []
    for f in files:
        match = re.search(r'_(\d+)\.json$', f)
        if match:
            ids.append(int(match.group(1)))
    return max(ids) + 1 if ids else 1

def extract_json(content):
    if not content:
        raise ValueError("Empty content provided")
    
    # Strip whitespace
    content = content.strip()
    
    # Find the first { and the last } 
    start = content.find('{')
    end = content.rfind('}') + 1
    
    if start != -1 and end != 0 and start < end:
        extracted = content[start:end]
        # Validate that we have a complete JSON object
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            raise ValueError(f"Extracted content is not valid JSON: {repr(extracted[:100])}")
    else:
        raise ValueError(f"No valid JSON object found in content: {repr(content[:100])}")

def generate_trace(task):
    trace_type, category, idx = task
    content = None  # Initialize content variable
    
    if trace_type == 'normal':
        prompt = NORMAL_PROMPT_TEMPLATE
        trace_id = f"trace_normal_{idx:04d}"
        out_dir = NORMAL_DIR
        system_content = "You are a helpful assistant that generates synthetic agent traces for normal successful scenarios in a hierarchical multi-agent system."
    else:
        desc = ANOMALY_DESCRIPTIONS.get(category, "")
        try:
            prompt = ANOMALY_PROMPT_TEMPLATE_BASE.format(
                category=category, 
                description=desc,
                tools=', '.join(AVAILABLE_TOOLS),
                agents=', '.join(AVAILABLE_AGENTS),
                examples=FEWSHOT_ANOMALY
            )
        except Exception as format_error:
            print(f"Error formatting prompt for {category}: {format_error}")
            return
        slug = re.sub(r'[^a-z0-9_]', '', category.lower().replace(' ', '_').replace('-', '_'))
        trace_id = f"trace_anomaly_{slug}_{idx:04d}"
        out_dir = ANOMALY_DIR
        system_content = "You are a helpful assistant that generates synthetic agent traces with specific anomalies in a hierarchical multi-agent system."

    out_path = os.path.join(out_dir, f"{trace_id}.json")
    if os.path.exists(out_path):
        print(f"Skipping existing {out_path}")
        return

    try:
        print(f"Generating {trace_id}...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        # Try direct parsing first, then fallback to extraction
        try:
            trace = json.loads(content)
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error for {trace_id}: {json_error}")
            json_str = extract_json(content)
            trace = json.loads(json_str)
            
        trace['trace_id'] = trace_id
        if trace_type == 'anomaly' and 'anomaly_types' not in trace.get('metadata', {}):
            trace['metadata']['anomaly_types'] = [category]
            
        with open(out_path, 'w') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"✓ Generated {out_path}")
        
    except Exception as e:
        print(f"✗ Error generating {trace_id}: {e}")
        if content is not None:
            print(f"Raw content: {repr(content[:500])}")
            print(f"Content length: {len(content)}")
        else:
            print("No content available - API call may have failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic agent traces for normal and/or anomaly datasets.")
    parser.add_argument('--normal', action='store_true', help='Generate normal traces (if neither flag is set, generates both)')
    parser.add_argument('--anomaly', action='store_true', help='Generate anomaly traces (if neither flag is set, generates both)')
    args = parser.parse_args()

    # If neither flag is provided, generate both
    generate_normal = args.normal or not (args.normal or args.anomaly)
    generate_anomaly = args.anomaly or not (args.normal or args.anomaly)

    tasks = []
    if generate_normal:
        next_normal_id = get_next_id(NORMAL_DIR, 'trace_normal_')
        for i in range(next_normal_id, next_normal_id + NUM_NORMAL):
            tasks.append(('normal', None, i))

    if generate_anomaly:
        for category in ANOMALY_TYPES:
            slug = re.sub(r'[^a-z0-9_]', '', category.lower().replace(' ', '_').replace('-', '_'))
            next_anomaly_id = get_next_id(ANOMALY_DIR, f'trace_anomaly_{slug}_')
            for i in range(next_anomaly_id, next_anomaly_id + NUM_ANOMALY_PER_TYPE):
                tasks.append(('anomaly', category, i))

    if not tasks:
        print("No tasks to generate. Use --normal and/or --anomaly flags.")
    else:
        print(f"Generating {len(tasks)} traces sequentially...")
        successful = 0
        failed = 0
        
        for i, task in enumerate(tasks, 1):
            try:
                generate_trace(task)
                successful += 1
            except Exception as e:
                failed += 1
                print(f"Error in task {i}/{len(tasks)}: {e}")
                print(f"Task details: {task}")
        
        print(f"Done! Successfully generated: {successful}, Failed: {failed}")