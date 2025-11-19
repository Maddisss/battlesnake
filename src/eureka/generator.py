from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

def build_prompt(task_desc, feedback):
    return f"""
You are designing a reward function for a Battlesnake RL agent.

Battlesnake environment:
{task_desc}

Performance feedback from prior trials:
{feedback}

Write a Python function:
def compute_reward(state, action, next_state):
    \"\"\"Return a scalar reward.\"\"\"
Return only valid Python code.
"""

def generate_reward_code(task_desc, feedback):
    prompt = build_prompt(task_desc, feedback)
    outputs = llm.generate([prompt], sampling)
    code = outputs[0].outputs[0].text
    return code