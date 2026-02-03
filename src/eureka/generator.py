from collections import Counter
import json
import random

import numpy as np
from battlesnake.src.eureka.prompts import get_summarize_last_steps_prompt
from battlesnake.src.llm.llm_client import LLMClient
from battlesnake.src.eureka.prompts import summarize_stats_prompt

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

    imports = """from agents.KILabAgentGroup3.AgentWrapper import AgentWrapper\nfrom environment.Battlesnake.model.Direction import Direction\nfrom environment.Battlesnake.model.Position import Position\n"""
    code_with_imports = imports + code

    try:
        exec(code_with_imports, {}, local_scope)
    except Exception as e:
        print("reward function generation error: {e}")
        return None

    if "compute_reward" not in local_scope:
        return None

    return local_scope["compute_reward"]


def summarize_training_stats(episode_stats, eureka_stats, seed=None):
    """
    Summarizes both episode-level and batch-level (Eureka) statistics
    and generates a concise textual summary plus direct reward recommendations
    for the LLM.
    """
    if seed is not None:
        random.seed(seed)

    # -------------------------
    # Episode-level stats
    # -------------------------
    n_games = len(episode_stats)
    if n_games == 0:
        return "No episode statistics available.", "No reward recommendations available."

    # Death reasons
    death_reasons = [ep["elimination_cause"] for ep in episode_stats]
    death_reason_counts = Counter(death_reasons)

    total_steps = sum(ep["steps_survived"] for ep in episode_stats)
    total_food = sum(ep["food_eaten"] for ep in episode_stats)
    total_enemies = sum(ep["enemies_killed"] for ep in episode_stats)

    avg_steps_survived = total_steps / n_games
    avg_enemies_killed = total_enemies / n_games
    avg_food_per_step = total_steps / total_food if total_food > 0 else 0.0
    wins = sum(1 for ep in episode_stats if ep["elimination_cause"] == "survived")
    win_rate = wins / n_games

    # -------------------------
    # Build concise episode summary
    # -------------------------
    episode_summary = [
        f"Total games: {n_games}",
        f"Win rate: {win_rate:.2%}",
        f"Average steps survived: {avg_steps_survived:.2f}",
        f"Average enemies killed per game: {avg_enemies_killed:.2f}",
        f"Average steps per food: {avg_food_per_step:.4f}",
        "Death reason counts:"
    ]
    for reason, count in death_reason_counts.items():
        episode_summary.append(f"  - {reason}: {count}")

    episode_summary_text = "\n".join(episode_summary)

    # -------------------------
    # Batch-level Eureka stats
    # -------------------------
    batch_summary_lines = []
    for i, batch in enumerate(eureka_stats):
        batch_summary_lines.append(f"\nBatch {i+1} (Step recorded: approx {batch.get('step_recorded', 0):.0f})")
        batch_summary_lines.append(f"  Win rate: {batch.get('win_rate', 0.0):.3f}")
        batch_summary_lines.append(f"  Average episode length: {batch.get('avg_length', 0.0):.1f}")
        batch_summary_lines.append("  Reward components:")
        for key in sorted(batch.get("means", {}).keys()):
            mean = batch["means"].get(key, 0.0)
            std = batch["stds"].get(key, 0.0)
            corr = batch["correlations"].get(key, 0.0)
            batch_summary_lines.append(f"    - {key}: mean={mean:.3f}, std={std:.3f}, corr_with_win={corr:.3f}")

    batch_summary_text = "\n".join(batch_summary_lines)

    # -------------------------
    # Generate direct reward shaping recommendations using LLM
    # -------------------------
    prompt = summarize_stats_prompt.format(episode_summary_text=episode_summary_text, batch_summary_text=batch_summary_text)

    reward_recommendation = prompt_llm(prompt)

    return episode_summary_text, reward_recommendation


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