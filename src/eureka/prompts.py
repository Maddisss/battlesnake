
import random


def get_random_generation_prompt():
    guidances = [basic_guidance, aggressive_guidance, defensive_guidance, balanced_guidance]
    guidance = random.choice(guidances)
    prompt = basic_generation_prompt.format(guidance=guidance)
    return prompt

def get_mutation_prompt(current_generation, generations, code, episode_stats, episode_rewards):
    if current_generation < generations / 2:
        return exploration_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards)
    else:
        return exploitation_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards)




game_rules_prompt = """
Core Battlesnake Rules

Objective:
- Survive longer than other snakes.
- Grow by eating food to increase length and health.
- Last snake alive wins (or rankings may consider length and survival in tournaments).

Board and Movement:
- Rectangular grid with fixed coordinates.
- Snakes move one square per turn: up, down, left, or right.
- Moves are simultaneous for all snakes.

Collisions:
- Hitting a wall or your own body → death.
- Head-to-body collision with another snake → death.
- Head-to-head: longer snake survives; equal length → both die.

Health and Food:
- Health decreases each turn; eating food restores it.
- Food also increases snake length by 1.

Game Flow:
- Server sends board state to each snake AI.
- Each AI returns a move.
- Server updates positions, health, and resolves collisions.
- Repeat until only one snake survives or the game times out.

Strategy Basics:
- Avoid collisions.
- Compete for food.
- Use positioning to block or trap opponents.
"""


basic_generation_prompt = """
You are an expert reinforcement learning engineer designing a reward function for a Battlesnake agent.

Environment:
- Board: 11x11 grid
- Player controls a snake; one enemy snake on the board
- Food randomly placed on empty cells
- Observation: 3-channel grid (snake body, head, food)
- Actions: 0=up, 1=down, 2=left, 3=right
- Maximum steps: 300
- Snake has health, decreasing over time if food not eaten
- Game ends if snake collides with wall, itself, or enemy head

Reward Function:
- Signature: def compute_reward(died, ate_food, killed_enemy, old_food_distance, new_food_distance, game_state) -> float
- game_state dictionary contains keys:

    'snake': [(r,c), ...],
    'enemy': [(r,c), ...],
    'food': (r,c),
    'board_size': int,
    'health': int,
    'step': int
  
- Use only the provided function arguments and game_state to compute rewards
- Must return a single scalar reward
- Keep rewards bounded between -10 and +10 to ensure stable training
- Do not perform I/O or modify the environment

{guidance}

Instructions:
- Provide only valid Python code implementing compute_reward
- Optional comments inside the function are allowed for clarity
- Do not include explanations or text outside the function
"""

basic_guidance = """
Guidance:
- Reward survival, efficient food collection, and intelligent combat
- Penalize death, collisions, risky head-to-head moves, and self-trapping
- Encourage open-space movement and avoiding traps
- Include intermediate rewards based on state and recent changes (distances, health, step)
- Reward function should adapt dynamically to the current game state; avoid static constant rewards
- You may create creative heuristics, as long as they are computable from the inputs
"""

defensive_guidance = """
Guidance:
- Prioritize survival above all else
- Penalize collisions, risky head-to-head moves, self-trapping, and dangerous positioning
- Reward staying in open spaces and safe navigation
- Reward food collection, but only if it doesn’t compromise safety
- Intermediate rewards should guide the snake to avoid traps and maintain health
- Dynamic reward: depend on current positions, distances, and health
"""

aggressive_guidance = """
Guidance:
- Prioritize killing the enemy whenever possible
- Reward moving towards the enemy and strategic positioning for attacks
- Penalize dying and collisions, but slightly less than the benefit of successful aggression
- Reward food collection to maintain health, but combat takes priority
- Penalize self-trapping or poor positioning that reduces combat opportunities
- Dynamic reward: consider enemy distance, health, and board control
"""

balanced_guidance = """
Guidance:
- Balance survival, food collection, and combat
- Reward staying alive, efficient food gathering, and defeating enemies
- Penalize self-trapping, collisions, and risky head-to-head moves
- Encourage open-space movement while taking advantage of combat opportunities
- Intermediate rewards should guide adaptive behavior depending on health, distance to food, and enemy proximity
- Dynamic reward: adapt to the current game state to favor long-term survival and strategic play
"""


exploration_based_mutation_prompt = """
You are given the following reward function for Battlesnake:

{reward_code}

You are also provided with performance statistics from several episodes:

Episode Stats:
{episode_stats}

Episode Rewards:
{episode_rewards}

Your task is to **mutate this reward function** to explore **new strategies and behaviors**. 
- Keep any parts that seem to correlate with good outcomes.
- Introduce new ideas to reward behaviors the snake has not consistently exhibited (e.g., alternative food strategies, positioning, risk-taking).
- You may add new terms, reshape existing rewards, or scale them differently.
- Avoid removing all elements that worked well; instead, combine exploration with retention.

Return **only Python code** for the mutated `compute_reward` function. 
Make it readable and self-contained.
"""

exploitation_based_mutation_prompt = """
You are given the following reward function for Battlesnake:

{reward_code}

You are also provided with performance statistics from several episodes:

Episode Stats:
{episode_stats}

Episode Rewards:
{episode_rewards}

Your task is to **mutate this reward function** to **exploit what worked well**:
- Identify which parts of the reward function correlate with high fitness.
- Increase the emphasis or weighting of these effective parts.
- Avoid introducing radically new behaviors; focus on improving and reinforcing success.
- Minor refinements, scaling, or reshaping of existing rewards are acceptable.

Return **only Python code** for the mutated `compute_reward` function.
Make it readable and self-contained.

"""
