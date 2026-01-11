import json
from src.llm.llm_client import LLMClient


def build_prompt(task_desc: str, feedback: str = None):
    feedback_section = f"Performance feedback from previous attempts:\n{feedback}\n" if feedback else ""
    
    return f"""
{task_desc}

{feedback_section}
"""


def generate_reward_code(task_desc, feedback: str = None):
    prompt = build_prompt(task_desc, feedback)

    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Respond ONLY with JSON containing the key 'code'."},
        {"role": "user", "content": prompt}
    ]

    json_str = LLMClient.chat(
        messages=messages,
        temperature=0.6,
        response_format={ "type": "json_object" }
    )
    data = json.loads(json_str)

    return data['code']


def load_reward_function(code: str):
    local_scope = {}

    try:
        exec(code, {}, local_scope)
    except Exception as e:
        raise RuntimeError(f"Reward code failed to compile: {e}")

    if "compute_reward" not in local_scope:
        raise ValueError("No compute_reward function found in LLM output")

    return local_scope["compute_reward"]