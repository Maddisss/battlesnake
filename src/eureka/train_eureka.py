

from collections import deque
import gc
import os
import time
import numpy as np
from agents.KILabAgentGroup3.callbacks.StatsCallback import BattlesnakeStatsWriterCallback
from agents.KILabAgentGroup3.utilities.environment_helper import vanilla_obs_to_readable_string
from battlesnake.src.envs.env_utilities import create_env
from battlesnake.src.eureka.eureka_db import EurekaDB
from battlesnake.src.eureka.generator import generate_reward_code, load_reward_function, summarize_episode_stats
from battlesnake.src.training.ppo import get_ppo_model, load_ppo_model
from battlesnake.src.eureka.prompts import get_mutation_prompt, get_random_generation_prompt


def continue_training(reward_fn, model_path, timesteps, n_eval_episodes=50):
    flashlight_mode = False
    env = create_env(reward_fn, flashlight_mode)

    if os.path.exists(model_path):
        model = load_ppo_model(model_path, env)
    else:
        model = get_ppo_model(env, flashlight_mode)

    stats_callback = BattlesnakeStatsWriterCallback()
    fitness = None

    try:

        # Train for the given timesteps
        model.learn(total_timesteps=timesteps, callback=stats_callback)
        model.save(model_path)

        # Evaluate
        episode_stats = []
        episode_rewards = []

        validate_env = create_env(reward_fn, flashlight_mode, evaluation_env=True)

        for _ in range(n_eval_episodes):
            obs, _ = validate_env.reset()
            done = False
            total_reward = 0.0
            food_eaten = 0
            enemies_killed = 0
            # last_vanilla_obs = deque(maxlen=10)

            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, trunc, info = validate_env.step(action)
                total_reward += reward
                # vanilla_obs = info['vanilla_obs']
                # if vanilla_obs:
                #     obs_string = vanilla_obs_to_readable_string(vanilla_obs)
                #     last_vanilla_obs.append(obs_string)

                stats = info['stats']

                food_eaten += 1 if stats.get("food_eaten", 0) == 1 else 0
                enemies_killed += 1 if stats.get("enemies_killed", 0) == 1 else 0
                if done or trunc:
                    episode_rewards.append(total_reward)
                    episode_stats.append({
                        "steps_survived": validate_env.step_count,
                        "food_eaten": food_eaten,
                        "enemies_killed": enemies_killed,
                        "elimination_cause": stats.get("elimination_cause", "survived"),
                        # "last_steps": list(last_vanilla_obs)
                    })


        fitness = calculate_fitness(episode_stats, episode_rewards)
    finally:
        if not fitness and not episode_rewards:
            fitness = -1
            episode_rewards = []
            episode_stats =[{}]
        env.close()
        if validate_env:
            validate_env.close()
            del validate_env
        model.env.close()
        del model.env
        del model, env
        gc.collect()
        time.sleep(0.2)
    return fitness, episode_stats, episode_rewards


def eureka_training_loop(generations=4, start_population=16):
    db = EurekaDB()
    population = start_population

    base_timesteps = 200000
    timesteps = base_timesteps
    parent_candidates = []
    for generation in range(generations):
        timesteps = base_timesteps * (2 ** generation)
        candidates = []
        # Population generation
        for i in range(population):
            parent = parent_candidates[i] if parent_candidates else None

            if parent:
                episode_stats_str = summarize_episode_stats(parent['episode_stats'])
                prompt = get_mutation_prompt(generation, generations, parent['code'], episode_stats_str, sum(parent['rewards']) / len(parent['rewards']))
                code = generate_reward_code(
                    prompt
                )
            else:
                generation_prompt = get_random_generation_prompt()
                code = generate_reward_code(generation_prompt)

            model_path = f"models/eureka_v2_candidate_g{generation}_{i}.zip"

            candidates.append({
                "code": code,
                "parent_id": parent['id'] if parent else None,
                "generation": generation,
                "model_path": model_path,
                "timesteps": timesteps
            })

        # Evaluation
        for candidate in candidates:
            fn = load_reward_function(candidate["code"])

            fitness, stats, rewards = continue_training(
                reward_fn=fn,
                model_path=candidate["model_path"],
                timesteps=timesteps
            )

            candidate["fitness"] = fitness
            candidate["rewards"] = rewards
            candidate["episode_stats"] = stats
            candidate["id"] = db.insert_candidate(
                generation=candidate["generation"],
                parent_id=candidate["parent_id"],
                code=candidate["code"],
                fitness=fitness,
                episode_stats=stats
            )
        
        # Selection
        candidates.sort(key=lambda c: c['fitness'], reverse=True)

        if len(candidates) == 1:
            print(f"Generation {generation} best fitness: {candidates[0]['fitness']:.4f}")
            return 

        survivors = candidates[: population // 2]

        parent_candidates = survivors
        population = len(parent_candidates)

        print(f"Generation {generation} best fitness: {survivors[0]['fitness']:.4f}")
        


def evaluate_candidate(model, env, n_episodes=50):
    stats_list = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, trunc, _ = env.step(action)
        stats_list.append({
            "steps_survived": env.steps_survived,
            "food_eaten": env.food_eaten,
            "enemies_killed": env.enemies_killed,
            "elimination_cause": env.elimination_cause
        })
    
    return stats_list



import numpy as np

def calculate_fitness(episode_stats, episode_rewards):
    """
    episode_stats: list of dicts with keys:
        - 'steps_survived'
        - 'food_eaten'
        - 'enemies_killed'
        - 'elimination_cause' (wall, self, enemy, hunger, etc.)
    episode_rewards: list of total reward per episode from the reward function
    
    Returns:
        fitness: float representing the fitness of the candidate
    """
    number_of_games = len(episode_rewards)

    # --- Extract behavior metrics ---
    steps = np.array([s['steps_survived'] for s in episode_stats])
    food = np.array([s['food_eaten'] for s in episode_stats])
    kills = np.array([s['enemies_killed'] for s in episode_stats])
    deaths = np.array([1 if s['elimination_cause'] and s['elimination_cause'] not in ['survived'] else 0 for s in episode_stats])

    wins = 0
    bonus = 0
    for i in range(number_of_games):
        wins += 1 if deaths[i] == 0 else 0
        # if death but survived more than 300 steps, give bonus:
        #  - 400 -> 0.1, 500 -> 0.2, 600 -> 0.3, >600 capped at 0.3
        if deaths[i] == 1 and steps[i] >= 100:
            extra_bonus = (steps[i] - 100) // 100
            bonus += min(0.1 + (0.1 * extra_bonus), 0.3)
        # if killed at least one enemy, give bonus
        if kills[i] >= 1:
            bonus += 0.25
        
    wins_norm = wins / number_of_games
    bonus_norm = bonus / number_of_games
    norm_fitness = wins_norm + bonus_norm

    return float(norm_fitness)


