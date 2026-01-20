
from collections import Counter
import random


def get_random_generation_prompt():
    guidances = [basic_guidance, aggressive_guidance, defensive_guidance, balanced_guidance]
    guidance = random.choice(guidances)
    prompt = basic_generation_prompt.format(guidance=guidance, code_context=code_context_prompt)
    return prompt

def get_mutation_prompt(current_generation, generations, code, episode_stats, episode_rewards):
    if current_generation < generations / 2:
        return exploration_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards, code_context=code_context_prompt)
    else:
        return exploitation_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards, code_context=code_context_prompt)


def get_summarize_last_steps_prompt():
    return sumarize_last_steps_prompt


sumarize_last_steps_prompt = """
You are analyzing the final steps of a Battlesnake game that ended in elimination.

Input

last_steps: A chronological list of (matrices with symbols to visualize the game board) game states from the final turns before elimination.

Your task

Identify the root cause of elimination

Determine what specific mistake or situation caused the snake to lose.

Focus on concrete failure modes such as:

self-collision

wall collision

head-to-head loss

starvation / health mismanagement

trap / no-escape scenario

poor space control

risky aggression

Base your conclusion strictly on the information visible in last_steps.

Short problem summary

In 1–2 sentences, clearly describe what went wrong and why it was inevitable or likely given the final positions.

Reward-function recommendations

Propose specific reward shaping changes for a reinforcement learning agent that would reduce the probability of this failure.

Recommendations must be actionable and framed as reward signals, for example:

penalties for entering low-freedom states

rewards for maintaining multiple escape paths

penalties for health dropping below a threshold without food access

discouraging head-to-head moves when length-disadvantaged

Avoid vague advice; each recommendation should describe:

what to measure

when it should trigger

whether it is a positive reward or penalty

Output format

Problem Summary:
<short explanation>

Likely Root Cause:
<primary failure mechanism>

Reward Function Improvements:
- <reward / penalty idea 1>
- <reward / penalty idea 2>
- <reward / penalty idea 3>


Constraints

Be short, concise and precise. 

Do not speculate beyond the given last_steps.

Assume the audience is designing a PPO-style reinforcement learning reward function.


The game board last steps in chronological order:
{last_steps}
"""

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

code_context_prompt = """
Reward Function:
- Signature: def compute_reward(agent: AgentWrapper, action: Direction, terminated: bool, action_is_safe: bool, ate_food: bool, died: bool, enemy_died: bool = False) -> float:
You may ONLY use the following inputs:

Scalars / Flags:
- terminated : bool          → episode ended this step
- died : bool                → agent died this step
- enemy_died : bool          → enemy snake died this step
- ate_food : bool            → agent ate food this step
- action_is_safe : bool      → chosen action was among safe moves

Agent State (read-only via agent.agent_context):
- agent.agent_context.you : Snake
    - you.body : List[Position]       (body[0] is head)
    - you.health : int                (0–100)
    - you.get_length() : int
    - you.get_head(), you.get_tail() : Position
    - you.get_current_direction() : Direction

- agent.agent_context.board : BoardState
    - board.width, board.height : int
    - board.food : List[Food(Position)]
    - board.snakes : List[Snake]      (alive snakes only)
    - board.is_out_of_bounds(Position) : bool
    - board.is_occupied_by_snake(Position) : bool
    - board.is_occupied_by_food(Position) : bool

Agent (provides utilities):
- agent.distance_to_nearest_food() : int
- agent.distance_to_nearest_enemy_head() : int
- agent.distance_to_nearest_wall() : int
- agent.length_advantage() : int          (agent length - enemy length)
- agent.reachable_free_space() : int (range 0-225) reachable free space of you snake
- agent.enemy_reachable_free_space() : int (range 0-225) average reachable free space of the enemies

Position:
- Position(x, y) with integer coordinates
- position.x : int
- position.y : int

Direction(Enum):
- Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
- Direction.opposite() : Direction
- Direction.board_delta() : Tuple[int, int]  (delta x,y for this direction)

Important Constraints:
- Do NOT access observations, history, future states, or environment internals.
- Do NOT mutate any objects or perform I/O.
- Reward must be a single float bounded approximately in [-10, +10].
- Computation must be fast (no heavy search or long loops).
"""


basic_generation_prompt = """
You are an expert reinforcement learning engineer designing a reward function for a Battlesnake agent.
Design a dense reward function for a single-agent Battlesnake environment that encourages long-term survival, safe navigation, food acquisition, and winning against one enemy snake.

Episode Terminates When:
- The agent snake dies (collision with wall, itself, or enemy head/body, or out of health), or
- The game engine reports termination (only one snake remains).

Environment:
- Grid size: board.width × board.height (typically 15×15).
- One controlled snake ("me") and one enemy snake.
- Food spawns on empty cells; eating food restores health and increases length.
- Snake health decreases each step if no food is eaten.
- On head to head collisions, the longer snake survives; if equal length, both die.

{code_context}

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

Average Episode Rewards: {episode_rewards}

Your task is to **mutate this reward function** to explore **new strategies and behaviors**. 
- Keep any parts that seem to correlate with good outcomes.
- Introduce new ideas to reward behaviors the snake has not consistently exhibited (e.g., alternative food strategies, positioning, risk-taking).
- You may add new terms, reshape existing rewards, or scale them differently.
- Avoid removing all elements that worked well; instead, combine exploration with retention.

{code_context}

Instructions:
- Provide only valid Python code implementing compute_reward
- Optional comments inside the function are allowed for clarity
- Do not include explanations or text outside the function

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

{code_context}

Instructions:
- Provide only valid Python code implementing compute_reward
- Optional comments inside the function are allowed for clarity
- Do not include explanations or text outside the function

Return **only Python code** for the mutated `compute_reward` function.
Make it readable and self-contained.

"""
