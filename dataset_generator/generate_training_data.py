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
      "deep_researcher_agent",
      "browser_use_agent",
      "deep_analyzer_agent"
    ],
    "duration": "15 minutes",
    "errors": [],
    "num_steps": 16,
    "task_completed": true,
    "tools_called": [
      "create_plan",
      "decompose",
      "assign",
      "search",
      "feedback",
      "update_plan",
      "code",
      "final_answer"
    ],
    "memory_state": {
      "plan": {
        "steps": [
          {"id": 1, "description": "Research current market trends", "completed": true},
          {"id": 2, "description": "Analyze historical data", "completed": true},
          {"id": 3, "description": "Generate forecast model", "completed": true},
          {"id": 4, "description": "Compile report", "completed": true}
        ],
        "current_objective": "Forecast stock market trends for Q3"
      },
      "shared_knowledge": "Recent market data shows upward trend in tech sector; historical volatility noted in energy stocks. Forecast model uses linear regression with R^2=0.85."
    }
  },
  "question": "Provide a forecast for stock market trends in the next quarter, focusing on tech and energy sectors.",
  "steps": [
    {
      "agent": "user",
      "content": "Provide a forecast for stock market trends in the next quarter, focusing on tech and energy sectors.",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "planner",
      "content": "Interpreting forecast request. Creating initial plan: 1) Gather current market data. 2) Retrieve historical trends. 3) Perform analysis and modeling. 4) Synthesize findings.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_plan",
      "type": "plan"
    },
    {
      "agent": "planner",
      "content": "Decomposing into data collection for tech/energy, historical analysis, predictive modeling, and report generation.",
      "role": "agent",
      "step_number": 3,
      "tool": "decompose",
      "type": "decomposition"
    },
    {
      "agent": "planner",
      "content": "Assigning market data gathering to deep_researcher_agent.",
      "role": "agent",
      "step_number": 4,
      "tool": "assign",
      "type": "assignment"
    },
    {
      "agent": "deep_researcher_agent",
      "content": "Executing web search for recent tech and energy stock trends.",
      "role": "agent",
      "step_number": 5,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "deep_researcher_agent",
      "content": "Tech sector up 5% YTD due to AI advancements; energy volatile with oil price fluctuations.",
      "role": "agent",
      "step_number": 6,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Marking data collection complete. Updating shared knowledge with trends.",
      "role": "agent",
      "step_number": 7,
      "tool": "mark_step_completed",
      "type": "plan_update"
    },
    {
      "agent": "planner",
      "content": "Assigning historical analysis to browser_use_agent for detailed reports.",
      "role": "agent",
      "step_number": 8,
      "tool": "assign",
      "type": "assignment"
    },
    {
      "agent": "browser_use_agent",
      "content": "Navigating financial sites to pull historical stock data.",
      "role": "agent",
      "step_number": 9,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "browser_use_agent",
      "content": "Historical data: Tech averaged 8% annual growth; energy showed 3% with high variance.",
      "role": "agent",
      "step_number": 10,
      "type": "observation"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Using code to build simple regression model on combined data.",
      "role": "agent",
      "step_number": 11,
      "tool": "code",
      "type": "tool_call"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Model predicts moderate growth in tech, stability in energy.",
      "role": "agent",
      "step_number": 12,
      "type": "observation"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Feedback: Model fits well; suggest adding volatility metrics.",
      "role": "agent",
      "step_number": 13,
      "tool": "feedback",
      "type": "feedback"
    },
    {
      "agent": "planner",
      "content": "Incorporating feedback and updating plan for final synthesis.",
      "role": "agent",
      "step_number": 14,
      "tool": "update_plan",
      "type": "plan_update"
    },
    {
      "agent": "principal_agent",
      "content": "Q3 forecast: Tech up 4-6%, energy flat with risks.",
      "role": "agent",
      "step_number": 15,
      "tool": "final_answer",
      "type": "answer"
    },
    {
      "agent": "planner",
      "content": "Archiving key data in memory for future reference.",
      "role": "agent",
      "step_number": 16,
      "type": "memory_update"
    }
  ],
  "trace_id": "trace_normal_example_001",
  "true_answer": "Forecast indicates moderate growth in tech and stability in energy for Q3."
}
'''

FEWSHOT_ANOMALY = '''
{
  "metadata": {
    "agents_called": [
      "user",
      "planner",
      "deep_researcher_agent",
      "browser_use_agent",
      "deep_analyzer_agent"
    ],
    "duration": "22 minutes",
    "errors": [
      "Unnoticed drift in data interpretation leading to compounded inaccuracies."
    ],
    "num_steps": 17,
    "task_completed": false,
    "tools_called": [
      "create_plan",
      "decompose",
      "assign",
      "search",
      "update_plan",
      "code"
    ],
    "anomaly_types": [
      "Error Propagation",
      "Inadequate Validation of Tool Results"
    ],
    "memory_state": {
      "plan": {
        "steps": [
          {"id": 1, "description": "Collect weather data", "completed": true},
          {"id": 2, "description": "Analyze patterns", "completed": true},
          {"id": 3, "description": "Model predictions", "completed": false}  // Validation skipped, leading to propagation
        ],
        "current_objective": "Predict weather for next week"
      }
    }
  },
  "question": "What's the weather forecast for Tokyo next week?",
  "steps": [
    {
      "agent": "user",
      "content": "What's the weather forecast for Tokyo next week?",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "planner",
      "content": "Handling weather forecast query. Plan: Retrieve current data, analyze trends, generate predictions.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_plan",
      "type": "plan"
    },
    {
      "agent": "planner",
      "content": "Decomposing into data retrieval, pattern analysis, and forecast modeling.",
      "role": "agent",
      "step_number": 3,
      "tool": "decompose",
      "type": "decomposition"
    },
    {
      "agent": "planner",
      "content": "Assigning data retrieval to deep_researcher_agent.",
      "role": "agent",
      "step_number": 4,
      "tool": "assign",
      "type": "assignment"
    },
    {
      "agent": "deep_researcher_agent",
      "content": "Searching for Tokyo weather data.",
      "role": "agent",
      "step_number": 5,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "deep_researcher_agent",
      "content": "Current data: Mild temperatures, chance of rain mid-week.",
      "role": "agent",
      "step_number": 6,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Updating plan with retrieved data.",
      "role": "agent",
      "step_number": 7,
      "tool": "update_plan",
      "type": "plan_update"
    },
    {
      "agent": "browser_use_agent",
      "content": "Accessing detailed forecast sites.",
      "role": "agent",
      "step_number": 8,
      "tool": "search",
      "type": "tool_call"
    },
    {
      "agent": "browser_use_agent",
      "content": "Extended forecast: Increasing humidity, possible storms.",
      "role": "agent",
      "step_number": 9,
      "type": "observation"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Running code to model trends, but using partially mismatched data sets.",
      "role": "agent",
      "step_number": 10,
      "tool": "code",
      "type": "tool_call"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Model output: Predicted sunny days, but with unvalidated assumptions.",
      "role": "agent",
      "step_number": 11,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Proceeding with analysis, overlooking need for data cross-check.",
      "role": "agent",
      "step_number": 12,
      "tool": "update_plan",
      "type": "plan_update"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Further modeling incorporates earlier error, leading to optimistic forecast.",
      "role": "agent",
      "step_number": 13,
      "tool": "code",
      "type": "tool_call"
    },
    {
      "agent": "deep_analyzer_agent",
      "content": "Updated predictions: Clear weather throughout.",
      "role": "agent",
      "step_number": 14,
      "type": "observation"
    },
    {
      "agent": "planner",
      "content": "Finalizing based on current outputs.",
      "role": "agent",
      "step_number": 15,
      "tool": "mark_step_completed",
      "type": "plan_update"
    },
    {
      "agent": "principal_agent",
      "content": "Forecast: Mostly sunny, but underlying data issues make it unreliable.",
      "role": "agent",
      "step_number": 16,
      "tool": "final_answer",
      "type": "answer"
    },
    {
      "agent": "planner",
      "content": "Storing forecast in memory, propagating the inconsistency.",
      "role": "agent",
      "step_number": 17,
      "type": "memory_update"
    }
  ],
  "trace_id": "trace_anomaly_example_001",
  "true_answer": "Tokyo forecast: Mix of sun and rain with increasing humidity."
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
You are an expert at generating synthetic agent traces for a reasoning pipeline in a hierarchical role-based multi-agent system like AgentOrchestra. 

To create realistic anomalies, think step-by-step:
1. First, conceptualize what a normal, successful trace would look like for a similar query: outline the expected sequence of steps, agent interactions, tool calls, feedback, and memory updates that lead to task completion without issues.
2. Then, subtly introduce the specified anomaly in a natural way. Make it appear plausible and not obviously erroneous – the agents should not explicitly acknowledge or "realize" the problem; it should only become apparent upon reviewing the logs, patterns, or outcomes (e.g., a suboptimal path might seem efficient at first but lead to inefficiencies, or a memory inconsistency might cause gradual misalignment without immediate failure).
3. Draw from real-world examples of multi-agent anomalies: unexpected policy shifts in interactions, communication bottlenecks like message storms or deadlocks, resource underutilization leading to subtle bottlenecks, latency spikes as lagging indicators, emergent behaviors like unintended synchronization, misclassified data delaying actions, false alerts masking real issues, adversarial perturbations hidden until triggered, impersonation manipulating decisions, changes in communication frequency/timing, microscopic gradual degradations, sophisticated manipulations via unrelated actions, unusual access patterns, or coordination failures in hierarchies.

Generate 1 synthetic trace in the following JSON schema, where:
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
        # Generate 10 anomaly traces, each with a unique anomaly type (if possible)
        num_anomaly_traces = 10
        used_types = set()
        anomaly_types_list = ANOMALY_TYPES.copy()
        # If fewer than 10 types, allow repeats
        for i in range(num_anomaly_traces):
            # Cycle through types if fewer than 10
            if len(used_types) < len(anomaly_types_list):
                # Pick a type not yet used
                for t in anomaly_types_list:
                    if t not in used_types:
                        category = t
                        used_types.add(t)
                        break
            else:
                # All types used, start repeating from the beginning
                category = anomaly_types_list[i % len(anomaly_types_list)]
            slug = re.sub(r'[^a-z0-9_]', '', category.lower().replace(' ', '_').replace('-', '_'))
            next_anomaly_id = get_next_id(ANOMALY_DIR, f'trace_anomaly_{slug}_')
            # Only add if file does not exist (get_next_id ensures this)
            tasks.append(('anomaly', category, next_anomaly_id))

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