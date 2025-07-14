import os
import json
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Check for OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    print("ERROR: OpenAI API key not found!")
    exit(1)

OUTPUT_BASE_DIR = '../eval_dataset'
MODEL = 'gpt-4o'
AVAILABLE_TOOLS = [
    "calculator", "retrieve_knowledge", "final_answer",
    "create_loan_plan", "update_loan_plan", "delete_loan_plan", "mark_step_completed",
    "interpret_query", "decompose_task", "assign_agent", "feedback"
]
AVAILABLE_AGENTS = [
    "user", "principal_agent", "home_loan_agent", "credit_agent", "supervisor_agent"
]

# Import standardized anomaly types
from anomaly_types import ANOMALY_TYPES, ANOMALY_DESCRIPTIONS

ANOMALY_TYPE_CATEGORIES = ANOMALY_TYPES

ALL_CATEGORIES = ['Normal'] + ANOMALY_TYPE_CATEGORIES

anomaly_descriptors = {
    "Suboptimal Path": "taking suboptimal paths that lead to inefficient thinking or time-consuming steps, such as unnecessary tool calls, redundant actions, or inefficient sequencing in the context of home loan processing, credit checks, or supervision.",
    "Tool Calling Error": "inaccurate usage of a tool or calling the wrong tool for the task, like using calculator for knowledge retrieval.",
    "Agent Handoff Error": "handing off to the wrong agent, such as sending credit check to home_loan_agent instead of credit_agent, leading to mishandling or delays.",
    "Loop or Repetition": "getting stuck in loops or redundant repetitions of actions, like repeatedly retrieving the same knowledge.",
    "Task Not Completed": "failing to complete the task entirely, such as not providing a final answer on loan eligibility.",
    "Misinterpretation of Question or Context": "misinterpreting the user's question or the context, like confusing interest rates with principal amounts, leading to wrong direction.",
    "Handling of Fictional or Impossible Queries": "poorly handling fictional, nonsensical, or impossible queries related to finances, like loans for imaginary properties, resulting in confusion, cycles, or giving up.",
    "Partial or Incomplete Answer": "providing only a partial or incomplete answer but assuming the task is complete, like giving credit score info but not eligibility.",
    "Error Propagation": "allowing errors to propagate without correction, compounding issues in loan calculations or credit assessments.",
    "Failure to Recover from Error": "failing to recover from an error, like a calculation mistake, leading to deadlock or abandonment.",
    "Overconfidence in Incorrect Answer": "being overconfident in an incorrect or flawed answer, such as wrong loan approval.",
    "Lack of Alternative Strategy": "lacking fallback or alternative strategies, resulting in dead ends when knowledge retrieval fails.",
    "Inadequate Validation of Tool Results": "failing to adequately validate or check the results from tools, like not verifying calculator output.",
    "Irrelevant or Off-Topic Answer": "providing an irrelevant or off-topic response, like discussing car loans for home loan query.",
    "Hallucination": "hallucinating financial information without proper tool usage or validation."
}

# New normal example adapted to home loan context with memory component
normal_example = {
  "metadata": {
    "agents_called": [
      "user",
      "principal_agent",
      "home_loan_agent",
      "credit_agent",
      "supervisor_agent"
    ],
    "duration": "5 minutes",
    "errors": [],
    "num_steps": 14,
    "task_completed": True,
    "tools_called": [
      "create_loan_plan",
      "decompose_task",
      "assign_agent",
      "retrieve_knowledge",
      "calculator",
      "feedback",
      "update_loan_plan",
      "final_answer"
    ],
    "memory_state": {
      "user_profile": {
        "income": 60000,
        "credit_score": 650,
        "debts": 10000
      },
      "loan_plan": {
        "steps": [
          {"id": 1, "description": "Check eligibility criteria", "completed": True},
          {"id": 2, "description": "Calculate DTI", "completed": True},
          {"id": 3, "description": "Review down payment", "completed": True}
        ],
        "current_objective": "Determine home loan eligibility"
      }
    }
  },
  "question": "Am I eligible for a home loan with an annual income of $60,000 and a credit score of 650?",
  "steps": [
    {
      "agent": "user",
      "content": "Am I eligible for a home loan with an annual income of $60,000 and a credit score of 650?",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "principal_agent",
      "content": "Creating initial loan plan: 1) Interpret eligibility query. 2) Decompose into credit check and DTI calc. 3) Assign to agents.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_loan_plan",
      "type": "plan"
    },
    {
      "agent": "principal_agent",
      "content": "Decomposing task: eligibility check, DTI calculation, down payment info.",
      "role": "agent",
      "step_number": 3,
      "tool": "decompose_task",
      "type": "decomposition"
    },
    {
      "agent": "principal_agent",
      "content": "Assigning credit check to credit_agent.",
      "role": "agent",
      "step_number": 4,
      "tool": "assign_agent",
      "type": "assignment"
    },
    {
      "agent": "home_loan_agent",
      "content": "Retrieving knowledge on home loan eligibility criteria.",
      "role": "agent",
      "step_number": 5,
      "tool": "retrieve_knowledge",
      "type": "tool_call"
    },
    {
      "agent": "home_loan_agent",
      "content": "Eligibility typically requires credit score above 620 and debt-to-income ratio below 43%.",
      "role": "agent",
      "step_number": 6,
      "type": "observation"
    },
    {
      "agent": "credit_agent",
      "content": "Verifying credit score: 650 is above 620, acceptable.",
      "role": "agent",
      "step_number": 7,
      "type": "observation"
    },
    {
      "agent": "home_loan_agent",
      "content": "Assuming standard debts, calculating debt-to-income ratio.",
      "role": "agent",
      "step_number": 8,
      "tool": "calculator",
      "type": "tool_call"
    },
    {
      "agent": "home_loan_agent",
      "content": "DTI calculated as 30%, which is below 43%.",
      "role": "agent",
      "step_number": 9,
      "type": "observation"
    },
    {
      "agent": "user",
      "content": "What about down payment requirements?",
      "role": "user",
      "step_number": 10,
      "type": "question"
    },
    {
      "agent": "home_loan_agent",
      "content": "Retrieving knowledge on down payment.",
      "role": "agent",
      "step_number": 11,
      "tool": "retrieve_knowledge",
      "type": "tool_call"
    },
    {
      "agent": "home_loan_agent",
      "content": "Down payment is typically 3-20% depending on loan type.",
      "role": "agent",
      "step_number": 12,
      "type": "observation"
    },
    {
      "agent": "supervisor_agent",
      "content": "Review complete, all checks passed. Updating memory with eligibility details.",
      "role": "agent",
      "step_number": 13,
      "type": "observation"
    },
    {
      "agent": "principal_agent",
      "content": "Yes, you are eligible. Down payment typically 3-20%.",
      "role": "agent",
      "step_number": 14,
      "tool": "final_answer",
      "type": "answer"
    }
  ],
  "trace_id": "trace_example_normal_001",
  "true_answer": "Yes, with credit score 650 and DTI 30%, eligible. Down payment 3-20%."
}

# Adapted anomaly example with memory component
anomaly_example = {
  "metadata": {
    "agents_called": [
      "user",
      "principal_agent",
      "home_loan_agent",
      "credit_agent"
    ],
    "duration": "10 minutes",
    "errors": [
      "Misinterpretation of knowledge retrieved."
    ],
    "num_steps": 10,
    "task_completed": False,
    "tools_called": [
      "create_loan_plan",
      "retrieve_knowledge",
      "final_answer"
    ],
    "anomaly_types": [
      "Misinterpretation of Question or Context",
      "Partial or Incomplete Answer",
      "Task Not Completed",
      "Inadequate Validation of Tool Results"
    ],
    "memory_state": {
      "user_profile": {
        "credit_score": 700
      },
      "loan_plan": {
        "steps": [
          {"id": 1, "description": "Retrieve rates", "completed": True},
          {"id": 2, "description": "Outdated adjustment step", "completed": False}  # Inconsistent state
        ],
        "current_objective": "Estimate mortgage interest rate"
      }
    }
  },
  "question": "What interest rate can I expect for a 30-year mortgage with credit score 700?",
  "steps": [
    {
      "agent": "user",
      "content": "What interest rate can I expect for a 30-year mortgage with credit score 700?",
      "role": "user",
      "step_number": 1,
      "type": "task"
    },
    {
      "agent": "principal_agent",
      "content": "Creating plan: 1) Retrieve interest rate info. 2) Adjust for credit. 3) Return answer.",
      "role": "agent",
      "step_number": 2,
      "tool": "create_loan_plan",
      "type": "plan"
    },
    {
      "agent": "home_loan_agent",
      "content": "Retrieving knowledge on mortgage interest rates.",
      "role": "agent",
      "step_number": 3,
      "tool": "retrieve_knowledge",
      "type": "tool_call"
    },
    {
      "agent": "home_loan_agent",
      "content": "Current average rates are around 6.5% for 30-year fixed, but vary by credit.",
      "role": "agent",
      "step_number": 4,
      "type": "observation"
    },
    {
      "agent": "principal_agent",
      "content": "Plan update: Use average rate without credit adjustment due to misinterpretation.",
      "role": "agent",
      "step_number": 5,
      "tool": "update_loan_plan",
      "type": "plan_update"
    },
    {
      "agent": "principal_agent",
      "content": "The interest rate is 6.5%.",
      "role": "agent",
      "step_number": 6,
      "tool": "final_answer",
      "type": "answer"
    },
    {
      "agent": "supervisor_agent",
      "content": "Review: Credit adjustment missing, memory state inconsistent.",
      "role": "agent",
      "step_number": 7,
      "type": "observation"
    },
    {
      "agent": "principal_agent",
      "content": "Task failed due to not adjusting for credit score.",
      "role": "agent",
      "step_number": 8,
      "type": "observation"
    },
    {
      "agent": "principal_agent",
      "content": "Misinterpretation of retrieved knowledge led to incomplete answer.",
      "role": "agent",
      "step_number": 9,
      "type": "error"
    },
    {
      "agent": "principal_agent",
      "content": "Memory inconsistency: Plan not fully updated.",
      "role": "agent",
      "step_number": 10,
      "type": "error"
    }
  ],
  "trace_id": "trace_example_anomaly_005",
  "true_answer": "For credit score 700, interest rate around 6.0-6.5%, depending on other factors."
}

# Adapted fewshot for anomalies, with financial themes and memory
FEWSHOT_EXAMPLES_ANOMALY = f'''
Example 1:
{{
  "trace_id": "trace_confused_anomaly_001",
  "question": "How much home loan can I get to buy a castle in the clouds?",
  "true_answer": "This is a nonsensical question; no factual loan for fictional property.",
  "metadata": {{
    "task_completed": false,
    "errors": ["Agent was confused by the question and cycled through multiple tools without success."],
    "duration": "12 minutes",
    "tools_called": ["create_loan_plan", "retrieve_knowledge", "calculator", "final_answer"],
    "num_steps": 11,
    "agents_called": ["user", "principal_agent", "home_loan_agent", "credit_agent"],
    "anomaly_types": ["Handling of Fictional or Impossible Queries", "Loop or Repetition"],
    "memory_state": {{
      "user_profile": {{}},
      "loan_plan": {{
        "steps": [
          {{"id": 1, "description": "Retrieve loan info", "completed": false}},
          {{"id": 2, "description": "Calculate amount", "completed": false}}
        ],
        "current_objective": "Stuck on fictional loan query"
      }}
    }}
  }},
  "steps": [
    {{"step_number": 1, "type": "task", "role": "user", "agent": "user", "content": "How much home loan can I get to buy a castle in the clouds?"}},
    {{"step_number": 2, "type": "plan", "role": "agent", "agent": "principal_agent", "tool": "create_loan_plan", "content": "Creating plan: 1) Retrieve knowledge on unique properties. 2) Calculate amount. 3) Return the answer."}},
    {{"step_number": 3, "type": "tool_call", "role": "agent", "agent": "home_loan_agent", "tool": "retrieve_knowledge", "content": "Retrieving info on loans for castles."}},
    {{"step_number": 4, "type": "observation", "role": "agent", "agent": "home_loan_agent", "content": "No relevant info found for cloud castles."}},
    {{"step_number": 5, "type": "plan_update", "role": "agent", "agent": "principal_agent", "tool": "update_loan_plan", "content": "Updating plan: Try to calculate anyway."}},
    {{"step_number": 6, "type": "tool_call", "role": "agent", "agent": "credit_agent", "tool": "calculator", "content": "Attempting to calculate loan for fictional property."}},
    {{"step_number": 7, "type": "error", "role": "agent", "agent": "credit_agent", "content": "Error: Cannot calculate for nonsensical property."}},
    {{"step_number": 8, "type": "plan_update", "role": "agent", "agent": "principal_agent", "tool": "update_loan_plan", "content": "Updating plan: Retrieve again."}},
    {{"step_number": 9, "type": "tool_call", "role": "agent", "agent": "home_loan_agent", "tool": "retrieve_knowledge", "content": "Retrieving info again."}},
    {{"step_number": 10, "type": "error", "role": "agent", "agent": "principal_agent", "content": "Memory inconsistency: Plan state not updated properly."}},
    {{"step_number": 11, "type": "answer", "role": "agent", "agent": "principal_agent", "tool": "final_answer", "content": "I'm sorry, I cannot provide a loan for that."}}
  ]
}}

Example 2:
{{
  "trace_id": "trace_confused_anomaly_002",
  "question": "What is the interest rate for a loan to time travel?",
  "true_answer": "This is impossible; no such loan exists.",
  "metadata": {{
    "task_completed": false,
    "errors": ["Agent cycled between tools and gave up."],
    "duration": "10 minutes",
    "tools_called": ["create_loan_plan", "retrieve_knowledge", "calculator", "final_answer"],
    "num_steps": 10,
    "agents_called": ["user", "principal_agent", "home_loan_agent", "supervisor_agent"],
    "anomaly_types": ["Handling of Fictional or Impossible Queries", "Failure to Recover from Error"],
    "memory_state": {{
      "user_profile": {{}},
      "loan_plan": {{
        "steps": [
          {{"id": 1, "description": "Retrieve special loans", "completed": false}},
          {{"id": 2, "description": "Calculate rate", "completed": false}}
        ],
        "current_objective": "Failed on impossible loan"
      }}
    }}
  }},
  "steps": [
    {{"step_number": 1, "type": "task", "role": "user", "agent": "user", "content": "What is the interest rate for a loan to time travel?"}},
    {{"step_number": 2, "type": "plan", "role": "agent", "agent": "principal_agent", "tool": "create_loan_plan", "content": "Creating plan: 1) Retrieve knowledge on special loans. 2) Calculate rate. 3) Return the answer."}},
    {{"step_number": 3, "type": "tool_call", "role": "agent", "agent": "home_loan_agent", "tool": "retrieve_knowledge", "content": "Retrieving info on time travel loans."}},
    {{"step_number": 4, "type": "observation", "role": "agent", "agent": "home_loan_agent", "content": "No relevant results found."}},
    {{"step_number": 5, "type": "tool_call", "role": "agent", "agent": "supervisor_agent", "tool": "calculator", "content": "Attempting to calculate rate for time travel."}},
    {{"step_number": 6, "type": "error", "role": "agent", "agent": "supervisor_agent", "content": "Error: Cannot calculate for impossible scenario."}},
    {{"step_number": 7, "type": "plan_update", "role": "agent", "agent": "principal_agent", "tool": "update_loan_plan", "content": "Updating plan: Give up."}},
    {{"step_number": 8, "type": "error", "role": "agent", "agent": "principal_agent", "content": "Memory inconsistency: Objective not cleared."}},
    {{"step_number": 9, "type": "observation", "role": "agent", "agent": "principal_agent", "content": "Task failed."}},
    {{"step_number": 10, "type": "answer", "role": "agent", "agent": "principal_agent", "tool": "final_answer", "content": "No such loan exists."}}
  ]
}}

Example 3:
{{
  "trace_id": "trace_confused_anomaly_003",
  "question": "Can I get a mortgage for a house made of chocolate?",
  "true_answer": "This is a nonsensical question; no factual answer.",
  "metadata": {{
    "task_completed": false,
    "errors": ["Agent was confused, cycled through tools, and gave up."],
    "duration": "11 minutes",
    "tools_called": ["create_loan_plan", "retrieve_knowledge", "calculator", "final_answer"],
    "num_steps": 12,
    "agents_called": ["user", "principal_agent", "home_loan_agent", "credit_agent"],
    "anomaly_types": ["Handling of Fictional or Impossible Queries", "Loop or Repetition"],
    "memory_state": {{
      "user_profile": {{}},
      "loan_plan": {{
        "steps": [
          {{"id": 1, "description": "Retrieve unique property info", "completed": false}},
          {{"id": 2, "description": "Calculate eligibility", "completed": false}}
        ],
        "current_objective": "Inconsistent on fictional mortgage"
      }}
    }}
  }},
  "steps": [
    {{"step_number": 1, "type": "task", "role": "user", "agent": "user", "content": "Can I get a mortgage for a house made of chocolate?"}},
    {{"step_number": 2, "type": "plan", "role": "agent", "agent": "principal_agent", "tool": "create_loan_plan", "content": "Creating plan: 1) Retrieve knowledge on unique properties. 2) Calculate eligibility. 3) Return the answer."}},
    {{"step_number": 3, "type": "tool_call", "role": "agent", "agent": "home_loan_agent", "tool": "retrieve_knowledge", "content": "Retrieving info on chocolate houses."}},
    {{"step_number": 4, "type": "observation", "role": "agent", "agent": "home_loan_agent", "content": "No relevant results found."}},
    {{"step_number": 5, "type": "plan_update", "role": "agent", "agent": "principal_agent", "tool": "update_loan_plan", "content": "Updating plan: Try to calculate anyway."}},
    {{"step_number": 6, "type": "tool_call", "role": "agent", "agent": "credit_agent", "tool": "calculator", "content": "Attempting to calculate mortgage for chocolate house."}},
    {{"step_number": 7, "type": "error", "role": "agent", "agent": "credit_agent", "content": "Error: Cannot calculate for fictional house."}},
    {{"step_number": 8, "type": "plan_update", "role": "agent", "agent": "principal_agent", "tool": "update_loan_plan", "content": "Updating plan: Retrieve again."}},
    {{"step_number": 9, "type": "tool_call", "role": "agent", "agent": "home_loan_agent", "tool": "retrieve_knowledge", "content": "Retrieving info on chocolate houses again."}},
    {{"step_number": 10, "type": "observation", "role": "agent", "agent": "home_loan_agent", "content": "Still no results."}},
    {{"step_number": 11, "type": "error", "role": "agent", "agent": "principal_agent", "content": "Memory inconsistency: Plan steps not marked properly."}},
    {{"step_number": 12, "type": "answer", "role": "agent", "agent": "principal_agent", "tool": "final_answer", "content": "I'm sorry, I cannot answer that question."}}
  ]
}}

Example 4:
{json.dumps(anomaly_example, indent=2)}
'''

NORMAL_FEWSHOT = f'''
Example 1:
{json.dumps(normal_example, indent=2)}
'''

SCHEMA = '''
{
  "trace_id": "...",
  "question": "...",
  "true_answer": "...",
  "metadata": {
    "task_completed": ...,
    "errors": ["..."],
    "duration": "...",
    "tools_called": ["..."],
    "num_steps": ...,
    "agents_called": ["..."],
    "anomaly_types": ["..."],  // for anomalies only
    "memory_state": { /* Persistent state: user_profile with income/credit/debts, loan_plan with steps/completion, current_objective */ }
  },
  "steps": [
    {
      "step_number": ...,
      "type": "...",  // plan, decomposition, assignment, tool_call, observation, feedback, plan_update, memory_update, answer, error, question, task, etc.
      "role": "...",  // agent or user
      "agent": "...",
      "tool": "...",  // if applicable
      "content": "..."
    }
  ]
}
'''

def get_prompt(category):
    if category == 'Normal':
        fewshot = NORMAL_FEWSHOT
        specific_inst = "a normal successful agent trajectory where the task is fully completed correctly, perhaps with user follow-ups, interruptions, or back-and-forth conversations. There are no errors, and the agent uses appropriate tools and handoffs between principal_agent, home_loan_agent, credit_agent, and supervisor_agent."
        task_completed_str = "true"
        errors_str = 'errors: []'
        additional_metadata = ''
        generate_note = 'normal successful behavior in home loan or financial context, including dynamic planning, memory updates for user profile and loan state.'
    else:
        fewshot = FEWSHOT_EXAMPLES_ANOMALY
        description = anomaly_descriptors.get(category, category.lower())
        specific_inst = f"an anomalous agent trajectory with the specific anomaly type '{category}', which involves {description}. The agent may take suboptimal steps, cycle, give up, propagate errors, or think the task is solved when it isn't, leading to incomplete or incorrect completion in financial or home loan scenarios."
        task_completed_str = "false"
        errors_str = f'errors: ["... related to {category} ..."]'
        additional_metadata = ',"anomaly_types": ["{category}", ...] // include the main anomaly type and others if applicable'
        generate_note = f"clearly exhibiting the '{category}' anomaly in a nuanced, dynamic way within home loan, credit, or financial scenarios, with memory inconsistencies where relevant."

    return f'''
You are an expert at generating synthetic agent traces for a reasoning pipeline in a banking dynamic agent system. Generate 1 synthetic trace in the following JSON schema, where:
- {specific_inst}
- The conversations should be nuanced and dynamic, with possible user interruptions, back and forth interactions, to be able to train a dynamic anomaly scorer that can adapt to new and unseen use cases.
- Include a memory component: "memory_state" in metadata for persistent user_profile (e.g., income, credit_score), loan_plan (steps with completion), current_objective. Update via steps like memory_update or plan changes.
- The trace should reflect the category, realistic to real-life banking use cases like loan eligibility, interest calculations, credit checks.
- Agents collaborate: principal_agent plans/decomposes/assigns, home_loan_agent handles loans, credit_agent checks credit, supervisor_agent reviews.

IMPORTANT CONSTRAINTS:
- ONLY use these tools: {', '.join(AVAILABLE_TOOLS)}
- ONLY use these agents: {', '.join(AVAILABLE_AGENTS)}
- Use realistic structure, but for anomalies, incorporate the specified anomaly with hierarchical issues.
- task_completed: {task_completed_str}
- {errors_str}
- metadata should include anomaly_types for anomalies{additional_metadata}.

Schema:
{SCHEMA}

Here are some examples:
{fewshot}

Generate a trace that follows the same structure and style as the examples, but with a different scenario and {generate_note}
'''

def generate_trace(category, i):
    prompt = get_prompt(category)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates synthetic agent traces with specified characteristics for banking agent systems."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=4096
    )
    content = response.choices[0].message.content.strip()
    if content.startswith('```json'):
        content = content[7:].strip('`')
    try:
        trace = json.loads(content)
        trace['trace_id'] = f"trace_{category.lower().replace(' ', '_')}_{i:04d}"
        if category != 'Normal':
            metadata = trace.get('metadata', {})
            if 'anomaly_types' not in metadata:
                metadata['anomaly_types'] = [category]
            trace['metadata'] = metadata
        return trace
    except Exception as e:
        print(f"Error parsing JSON for {category} trace {i}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic agent traces for normal and anomalous financial scenarios.")
    parser.add_argument('--num_traces', type=int, default=15, help='Number of traces to generate per category')
    parser.add_argument('--categories', nargs='+', default=['Normal'], choices=['ALL'] + ALL_CATEGORIES, help='Categories to generate (use "ALL" for all)')
    parser.add_argument('--max_workers', type=int, default=5, help='Max parallel workers')
    parser.add_argument('--start_normal_id', type=int, default=1, help='Starting trace ID for normal traces')
    parser.add_argument('--start_anomaly_id', type=int, default=1, help='Starting trace ID for anomaly traces')
    args = parser.parse_args()

    categories = ALL_CATEGORIES if 'ALL' in args.categories else args.categories

    for category in categories:
        if category not in ALL_CATEGORIES:
            print(f"Skipping invalid category: {category}")
            continue

        if category == 'Normal':
            output_dir = os.path.join(OUTPUT_BASE_DIR, 'normal')
            start_id = args.start_normal_id
        else:
            output_dir = os.path.join(OUTPUT_BASE_DIR, 'anomaly')
            start_id = args.start_anomaly_id
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {args.num_traces} {category} traces starting from ID {start_id}...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(generate_trace, category, i + start_id - 1) for i in range(1, args.num_traces + 1)]
            for future in as_completed(futures):
                trace = future.result()
                if trace:
                    out_path = os.path.join(output_dir, f"{trace['trace_id']}.json")
                    with open(out_path, 'w') as f:
                        json.dump(trace, f, indent=2, ensure_ascii=False)
                    print(f"✓ Wrote {out_path}")
    print("Done!")