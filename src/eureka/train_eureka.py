

import os
import numpy as np
from src.eureka.eureka_db import EurekaDB
from src.eureka.eureka_env import LLMRewardBattleSnake
from src.eureka.generator import generate_reward_code, load_reward_function
from src.training.ppo import get_ppo_model, load_ppo_model
from src.eureka.prompts import get_mutation_prompt, get_random_generation_prompt


def continue_training(reward_fn, model_path, timesteps, n_eval_episodes=50):
    env_fn = lambda: LLMRewardBattleSnake(reward_fn)

    if os.path.exists(model_path):
        model = load_ppo_model(model_path)
    else:
        model = get_ppo_model(env_fn)

    # Train for the given timesteps
    model.learn(total_timesteps=timesteps)
    model.save(model_path)

    # Evaluate
    episode_stats = []
    episode_rewards = []

    for _ in range(n_eval_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            if done or trunc:
                break

        episode_rewards.append(total_reward)
        episode_stats.append({
            "steps_survived": env.steps,
            "food_eaten": getattr(env, "food_eaten", 0),
            "enemies_killed": getattr(env, "enemies_killed", 0),
            "death_cause": getattr(env, "death_cause", "unknown")
        })


    fitness = calculate_fitness(episode_stats, episode_rewards)
    return fitness, episode_stats, episode_rewards


def eureka_training_loop(generations=4, population=16):
    db = EurekaDB()
    db.top_candidates(1)[0]

    base_timesteps = 50000
    timesteps = base_timesteps
    parent_candidates = []
    for generation in range(generations):
        candidates = []
        # Population generation
        for i in range(population):
            parent = parent_candidates[i] if parent_candidates else None

            if parent:
                prompt = get_mutation_prompt(generation, generations, parent['code'], parent['episode_stats'], parent['rewards'])
                code = generate_reward_code(
                    prompt
                )
            else:
                generation_prompt = get_random_generation_prompt()
                code = generate_reward_code(generation_prompt)

            model_path = f"models/eureka_candidate_g{generation}_{i}.zip"

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
                timesteps=base_timesteps
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
        timesteps = base_timesteps * (1 + generation)

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
            "death_cause": env.death_cause
        })
    
    return stats_list


import numpy as np

import numpy as np

def calculate_fitness(episode_stats, episode_rewards):
    """
    episode_stats: list of dicts with keys:
        - 'steps_survived'
        - 'food_eaten'
        - 'enemies_killed'
        - 'death_cause' (wall, self, enemy, hunger, etc.)
    episode_rewards: list of total reward per episode from the reward function
    
    Returns:
        fitness: float representing the fitness of the candidate
    """
    number_of_games = len(episode_rewards)

    # --- Extract behavior metrics ---
    steps = np.array([s['steps_survived'] for s in episode_stats])
    food = np.array([s['food_eaten'] for s in episode_stats])
    kills = np.array([s['enemies_killed'] for s in episode_stats])
    deaths = np.array([1 if s['death_cause'] in ['wall', 'self'] else 0 for s in episode_stats])

    wins = 0
    bonus = 0
    for i in range(number_of_games):
        win += 1 if deaths[i] == 0 else 0
        # if death but survived more than 300 steps, give bonus:
        #  - 400 -> 0.1, 500 -> 0.2, 600 -> 0.3, >600 capped at 0.3
        if deaths[i] == 1 and steps[i] >= 300:
            extra_bonus = (steps[i] - 300) // 100
            bonus += min(0.1 * extra_bonus, 0.3)
        # if killed at least one enemy, give bonus
        if kills[i] >= 1:
            bonus += 0.25
        
    wins_norm = wins / number_of_games
    bonus_norm = bonus / number_of_games
    norm_fitness = wins_norm + bonus_norm

    return float(norm_fitness)
