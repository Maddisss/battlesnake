from collections import Counter
import json
import random
from battlesnake.src.eureka.prompts import get_summarize_last_steps_prompt
from battlesnake.src.llm.llm_client import LLMClient


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

def prompt_llm(prompt):
    messages = [
        {"role": "system", "content": "You are an Advisor that analyzes Battlesnake games to recommend reward shaping improvements."},
        {"role": "user", "content": prompt}
    ]
    response = LLMClient.chat(
        messages=messages,
        temperature=0.6,
    )
    return response


def load_reward_function(code: str):
    local_scope = {}

    imports = """from agents.KILabAgentGroup3.AgentWrapper import AgentWrapper\nfrom environment.Battlesnake.model.Direction import Direction\n"""
    code_with_imports = imports + code

    try:
        exec(code_with_imports, {}, local_scope)
    except Exception as e:
        raise RuntimeError(f"Reward code failed to compile: {e}")

    if "compute_reward" not in local_scope:
        raise ValueError("No compute_reward function found in LLM output")

    return local_scope["compute_reward"]


def summarize_episode_stats(episode_stats, seed=None):
    if seed is not None:
        random.seed(seed)

    n_games = len(episode_stats)
    if n_games == 0:
        return "No episode statistics available."

    # -------------------------
    # Aggregate basic counters
    # -------------------------
    death_reasons = [ep["elimination_cause"] for ep in episode_stats]
    death_reason_counts = Counter(death_reasons)

    total_steps = sum(ep["steps_survived"] for ep in episode_stats)
    total_food = sum(ep["food_eaten"] for ep in episode_stats)
    total_enemies = sum(ep["enemies_killed"] for ep in episode_stats)

    # -------------------------
    # Derived metrics
    # -------------------------
    avg_food_per_step = (
        total_steps / total_food if total_food > 0 else 0.0
    )

    avg_enemies_killed = total_enemies / n_games
    avg_steps_survived = total_steps / n_games

    wins = sum(1 for ep in episode_stats if ep["elimination_cause"] == "survived")
    win_rate = wins / n_games

    # -------------------------
    # Select example games
    # -------------------------
    survived_games = [
        ep for ep in episode_stats if ep["elimination_cause"] == "survived"
    ]

    # survived_example = (
    #     random.choice(survived_games)["last_steps"]
    #     if survived_games else []
    # )

    # Find most common elimination cause (excluding survived)
    elimination_only = [
        r for r in death_reasons if r != "survived"
    ]

    # elimination_example = []
    # if elimination_only:
    #     most_common_elim, _ = Counter(elimination_only).most_common(1)[0]
    #     candidate_games = [
    #         ep for ep in episode_stats
    #         if ep["elimination_cause"] == most_common_elim
    #     ]
    #     elimination_example = random.choice(candidate_games)["last_steps"]

    # -------------------------
    # Build output string
    # -------------------------
    lines = []

    lines.append("Episode Summary")
    lines.append("-" * 40)

    lines.append("Death reason counts:")
    for reason, count in death_reason_counts.items():
        lines.append(f"  - {reason}: {count}")

    lines.append("")
    lines.append(f"Win rate: {win_rate:.2%}")
    lines.append(f"Average steps survived per game: {avg_steps_survived:.2f}")
    lines.append(f"Average enemies killed per game: {avg_enemies_killed:.2f}")
    lines.append(f"Average steps per food: {avg_food_per_step:.4f}")


    # lines.append("")
    # lines.append("Example game (eliminated – most common cause):")



    # if elimination_example:
    #     lines.append(get_last_steps_recommendation(elimination_example))
    # else:
    #     lines.append("  None available")

    return "\n".join(lines)


def get_last_steps_recommendation(elimination_example):
    example = '\n'.join([str(el) for el in elimination_example])
    last_steps_example = """
        "me_head": "H",
        "enemy_head": "E",
        "me_body": "B",
        "enemy_body": "b",
        "food": "F",
        "ground": ".",
        "unknown_ground": "?"
    """ + example
    sumarize_last_steps_prompt = get_summarize_last_steps_prompt()
    sumarize_last_steps_prompt.format(last_steps=last_steps_example)
    recommendation = prompt_llm(sumarize_last_steps_prompt)
    return recommendation