
from collections import Counter
import random


def get_random_generation_prompt():
    guidances = [aggressive_guidance, defensive_guidance, balanced_guidance]
    guidance = random.choice(guidances)
    prompt = basic_generation_prompt.format(guidance=guidance, code_context=code_context_prompt, reward_design_principles=reward_design_principles)
    return prompt

def get_mutation_prompt(current_generation, generations, code, episode_stats, episode_rewards):
    if current_generation < generations / 2:
        return exploration_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards, code_context=code_context_prompt)
    else:
        return exploitation_based_mutation_prompt.format(reward_code=code, episode_stats=episode_stats, episode_rewards=episode_rewards, code_context=code_context_prompt)

def get_offspring_prompt(code):
    prompt = basic_generation_prompt_based_on_existing_function.format(reward_code=code, code_context=code_context_prompt, reward_design_principles=reward_design_principles)
    return prompt

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

Signature:
def compute_reward(agent: AgentWrapper, action: Direction, terminated: bool, ate_food: bool, died: bool, enemy_died: bool = False) -> tuple[float, dict]:

You may ONLY use the inputs listed below.
- agent: AgentWrapper providing read-only access to the agent state and helper signals.
- action: Direction chosen by the agent this step
- terminated: episode ended this step
- ate_food: agent ate food this step
- died: agent died this step
- enemy_died: at least one enemy died this step


Return:
- total_reward: float (approx bounded in [-10, +10])
- components: dict[str, float] (reward_component and its reward contribution. Every used component must appear; unused set to 0)


==============================
AGENT STATE (read-only)
==============================
agent.agent_context.you:
- body[0] is head
- health in [0,100]
- get_length()
- get_head() : Position     Get current head position on the board get_head().x, get_head().y (the board is of size 11x11)
- get_tail()
- get_current_direction()

agent.agent_context.board:
- width, height
- turn
- food: list of positions
- snakes: alive snakes
- is_out_of_bounds(pos)
- is_occupied_by_snake(pos)
- is_occupied_by_food(pos)

==============================
HELPER SIGNALS (only call these functions ONCE and REUSE the returned values)
==============================
- agent.distance_to_nearest_food()     uses rounded Euclidean distance (not Manhattan). Returns board.width + board.height when no head or no food.
- agent.distance_to_nearest_enemy_head()     uses rounded Euclidean distance to closest enemy head; returns board.width + board.height if none.
- agent.reachable_free_space()    return a float “openness” score from a flood‑fill with exponential decay by distance in range (1-10) (not raw cell count). Can be fractional.
- agent.enemy_reachable_free_space()    return a float “openness” score from a flood‑fill with exponential decay by distance in range (1-10) (not raw cell count). Can be fractional.
- agent.reachable_free_space_from_action(action)    evaluates openness in range (1-10) from the next cell after taking action; returns 0 if action is a occupied cell.
- agent.length_advantage()    is my_length - max_alive_enemy_length. Positive means longer than the largest enemy.
- agent.get_shortest_path_to_snake_with_smallest_length_advantage()
    -> (enemy_snake, distance, path, length_advantage)

==============================
STORAGE
==============================
- agent.storage : dict is cleared each episode; use it for deltas (e.g., store previous distance/space advantage).

==============================
CONSTRAINTS
==============================
- No history access except via agent.storage
- No mutation of environment
- No heavy search / large loops
- Recommend smooth terms (tanh/clipped deltas) and explicit terminal reward.
- Reward must be smooth and stable (avoid extreme spikes except death/win)
- Try to be as computationally efficient as possible (returned values of functions should be reused.)
"""

reward_design_principles = """
Reward Design Principles:

1. Structure reward into:
   - Terminal rewards (win/loss)
   - Safety & survival shaping
   - Space control
   - Food & health management
   - Combat positioning

2. Terminal rewards (must dominate long-term objective):
   - Large negative for dying (e.g., -8 to -12)
   - Large positive for winning or eliminating final enemy (e.g., +8 to +12)
   - Terminal reward magnitude must exceed cumulative shaping over ~10 steps
   - Surviving longer should never be worse than dying immediately

3. Survival drift constraint (anti-collapse rule):
   - Expected per-step reward while alive should be near zero (roughly in [-0.2, +0.2])
   - Avoid strong constant negative pressure during normal survival
   - Do not create reward structures where dying early yields higher return than surviving

4. Dense shaping:
   - Use small incremental signals (0.05–1.0 range)
   - Prefer proportional rewards over binary bonuses
   - Avoid constant per-step rewards or penalties unless justified

5. Risk awareness:
   - Encourage maintaining or improving reachable space
   - Avoid large direct penalties for temporary space reduction
   - Penalize moving closer to stronger enemies
   - Reward space advantage over enemies

6. Phase adaptation:
   - Low health → gently increase food prioritization (scaled, not binary)
   - Length advantage → allow mild aggression shaping
   - Length disadvantage → mildly discourage head-to-head
   - Phase adjustments should reweight existing signals, not introduce large new penalties

7. Space control:
   - Reward relative space advantage over enemies
   - Prefer normalized differences over raw differences
   - Avoid absolute space rewards that drift negatively as board fills naturally

8. Food & health management:
    - Reward eating food
    - When health is low, gently reward decreasing distance to food
    - Avoid punishing food distance strongly when health is safe
   
9. Combat positioning:
    - Only reward aggression when clear length advantage exists
    - Scale enemy distance shaping proportionally
    - Avoid large proximity penalties unless immediate danger exists

10. Anti-hacking:
   - Avoid rewards that can be farmed by oscillation
   - Avoid encouraging stalling
   - Avoid rewarding unsafe but lucky behavior

11. Magnitude discipline:
   - Individual shaping terms typically in [-0.5, 0.5]
   - Avoid stacking more than ~1.0 total negative shaping in a single step
   - Death penalty should be significantly larger than any single-step shaping signal

12. Stability & consistency constraint:
   - Shaping rewards must be temporally consistent: similar states should yield similar rewards.
   - Avoid combining multiple high-sensitivity signals (e.g., space + wall + enemy distance) with large weights.
   - Reward gradients should be smooth and monotonic with respect to safety and advantage.
   - Small state changes must not cause large reward sign flips.
   - The agent should not experience strong negative reward while behaving safely and non-aggressively.
"""

basic_generation_prompt = """
You are an expert reinforcement learning engineer designing a reward function for a Battlesnake agent.
Design a dense reward function for a single-agent Battlesnake environment that encourages long-term survival, safe navigation, food acquisition, and winning against multiple enemy snakes.

Episode Terminates When:
- The agent snake dies (collision with wall, itself, or enemy head/body, or out of health), or
- The game engine reports termination (only one snake remains).

Environment:
- Grid size: board.width × board.height (typically 11×11).
- One controlled snake ("me") and multiple enemy snakes.
- Food spawns on empty cells; eating food restores health and increases length.
- Snake health decreases each step if no food is eaten.
- On head to head collisions, the longer snake survives; if equal length, both die.

{code_context}

{reward_design_principles}

{guidance}

Instructions:
- Provide only valid Python code implementing compute_reward
- Optional comments inside the function are allowed for clarity
- Do not include explanations or text outside the function

Before writing the final reward:
- Think about how each term affects long-term survival.
- Ensure no single shaping term can dominate terminal rewards.
- Ensure rewards do not encourage suicidal attacks.
"""

basic_generation_prompt_based_on_existing_function = """
You are an expert reinforcement learning engineer creating an improved offspring from an existing reward function for a Battlesnake agent.

Your task is to refine and strengthen the current reward design — NOT to redesign it from scratch. 
Preserve the overall structure and intent of the existing reward, but improve scaling, balance, stability, and long-term incentives where needed.

This is the current reward function code:
{reward_code}

The reward function must remain dense and encourage:
- Long-term survival
- Safe navigation
- Strategic food acquisition
- Competitive play against multiple enemy snakes

Episode Terminates When:
- The agent snake dies (collision with wall, itself, or enemy head/body, or out of health), or
- The game engine reports termination (only one snake remains).

Environment:
- Grid size: board.width × board.height (typically 11×11).
- One controlled snake ('me') and multiple enemy snakes.
- Food spawns on empty cells; eating food restores health and increases length.
- Snake health decreases each step if no food is eaten.
- On head-to-head collisions, the longer snake survives; if equal length, both die.

{code_context}

{reward_design_principles}

Improvement Constraints:
- Keep all existing major reward components unless clearly harmful.
- Adjust magnitudes, normalization, or conditioning instead of removing structure.
- Reduce unintended incentives (e.g., reckless aggression, reward hacking, oscillatory behavior).
- Ensure terminal rewards dominate shaping terms appropriately.
- Maintain smooth reward gradients where possible.
- Avoid sparse-only or purely binary reward redesigns.
- You can introduce new shaping ideas, but they must complement and integrate with the existing design.

Stability Requirements:
- No single shaping term should outweigh death or win rewards.
- Avoid incentives that encourage suicidal head-to-head attacks unless strongly favored.
- Ensure reward scales reasonably with board size and number of snakes.
- Avoid extremely large constants or unstable growth terms.

Instructions:
- Provide only valid Python code implementing compute_reward
- Optional comments inside the function are allowed for clarity
- Do not include explanations or text outside the function

Before writing the final reward:
- Review how each term contributes to long-term survival.
- Check for reward hacking opportunities.
- Ensure improvements refine the existing logic rather than replace it.
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

Primary objective: survive as long as possible.

TERMINAL:
- Very strong penalty for death
- Small bonus for outlasting enemies

SAFETY:
- Penalize unsafe actions heavily
- Penalize moves that reduce reachable space
- Penalize being near stronger enemy heads
- Penalize being near walls when space is limited

SPACE MAXIMIZATION:
- Reward high reachable_free_space
- Reward increasing space over time

FOOD:
- Reward eating food
- Reward decreasing food distance when health < 40
- Avoid risky food near enemies

COMBAT:
- Only reward aggression when strong length advantage exists
"""

aggressive_guidance = """
Guidance:

Primary objective: eliminate opponents while maintaining survival.

TERMINAL:
- Large reward for enemy_died
- Large penalty for died

COMBAT PRIORITY:
- If length_advantage > 0:
    - Reward decreasing distance to weaker enemy head
    - Reward reducing enemy reachable space
    - Reward space dominance (my_space - enemy_space)
- Encourage head-to-head positioning when advantaged

CONTROL:
- Reward central positioning when dominant
- Penalize giving enemy escape space

SURVIVAL:
- Penalize unsafe moves
- Penalize self-trapping strongly
- Low health still requires food prioritization

Aggression should scale with length_advantage and space advantage.
Avoid reckless suicide attacks.
"""

balanced_guidance = """
Guidance:

Primary objective: maximize win probability.

Encourage:

SURVIVAL:
- Strong penalty for death
- Penalize unsafe actions
- Penalize reduction in reachable space
- Reward maintaining high reachable_free_space

SPACE CONTROL:
- Reward having more reachable space than enemies
- Penalize being trapped or near walls when space is low

FOOD STRATEGY:
- Reward decreasing distance to food when health is low
- Reward eating food
- Penalize ignoring food when health is critical

COMBAT:
- If length_advantage > 0:
    - Reward decreasing distance to weaker enemy heads
    - Reward positioning that reduces enemy space
- If length_advantage < 0:
    - Penalize proximity to stronger enemy heads

ADAPTIVITY:
- Health < 30 → prioritize food
- Length advantage > 1 → increase aggression weight
- Low reachable space → increase safety penalty weight

Avoid static constant rewards.
Reward must reflect current board state.
"""



exploration_based_mutation_prompt = """
You are evolving a Battlesnake reward function.

==============================
CURRENT REWARD FUNCTION
==============================
{reward_code}

==============================
TRAINING DIAGNOSIS
==============================
{episode_stats}

Average episode reward: {episode_rewards}

==============================
OBJECTIVE
==============================
Introduce controlled strategic exploration while preserving proven useful components.

==============================
MUTATION RULES
==============================

1. Preserve components that:
   - Correlate positively with winning
   - Support survival and space control

2. Modify components that:
   - Show weak or negative correlation
   - Dominate total reward excessively

3. Introduce at most 2 new shaping ideas:
   - Alternative food urgency scaling
   - Space advantage comparison
   - Conditional aggression scaling
   - Nonlinear penalties (e.g., squared low-space penalty)
   - Health-based dynamic weighting

4. Do NOT:
   - Remove terminal rewards
   - Remove safety penalties
   - Rewrite everything from scratch
   - Add more than 1–2 new concepts

5. Maintain smooth reward scaling and stability.
6. Ensure no single shaping term exceeds terminal rewards.

==============================
CONSTRAINTS
==============================
{code_context}

Return ONLY valid Python code implementing compute_reward.
"""

exploitation_based_mutation_prompt = """
You are refining a Battlesnake reward function to exploit successful behaviors.

==============================
CURRENT REWARD FUNCTION
==============================
{reward_code}

==============================
TRAINING DIAGNOSIS
==============================
{episode_stats}

Average episode reward: {episode_rewards}

==============================
OBJECTIVE
==============================
Strengthen reward components that correlate with higher win rate.
Reduce or reshape components that harm performance.

==============================
REFINEMENT RULES
==============================

1. Increase weights of:
   - Positive corr_with_win components
   - Signals tied to survival and space dominance

2. Decrease or rescale:
   - Negative correlation components
   - Noisy high-variance components

3. Improve scaling:
   - Replace flat bonuses with proportional shaping
   - Make aggression depend more strongly on length_advantage
   - Make food urgency depend nonlinearly on health

4. Do NOT:
   - Introduce new unrelated reward concepts
   - Radically change structure
   - Remove core survival signals

5. Keep reward smooth and stable.

==============================
CONSTRAINTS
==============================
{code_context}

==============================
INSTRUCTIONS
==============================
Before writing the final function:
- Briefly reason about what specifically must change.
- Ensure the modification addresses the diagnosed failure.
- Ensure reward magnitude remains bounded.


Return ONLY valid Python code implementing compute_reward.
"""



summarize_stats_prompt = """
You are a reinforcement learning reward optimization expert for competitive multi-agent environments.

Your goal is to diagnose weaknesses in the current reward function and propose precise weight-level adjustments.

==============================
EPISODE-LEVEL PERFORMANCE
==============================
{episode_summary_text}

==============================
BATCH-LEVEL REWARD STATISTICS
For each reward component:
- mean: average contribution
- std: variability
- corr_with_win: correlation with win rate
==============================
{batch_summary_text}

==============================
ANALYSIS INSTRUCTIONS
==============================

1. Identify primary bottleneck limiting win rate:
   - Early deaths?
   - Starvation?
   - Losing head-to-head?
   - Poor space control?
   - Lack of aggression?

2. For each reward component:
   - If corr_with_win > 0.2 → likely beneficial
   - If corr_with_win < -0.2 → likely harmful
   - If near zero → weak signal

3. Detect pathologies:
   - High mean but negative correlation → misaligned shaping
   - High variance but zero correlation → noisy signal
   - Very large magnitude compared to others → dominance risk

4. Produce:
   A) Core failure diagnosis (short)
   B) Reward components to increase (with relative scaling suggestion)
   C) Reward components to decrease
   D) Missing strategic signals to introduce
   E) Risk warnings (reward hacking, oscillation, suicidal aggression)

Be concise and concrete.
Do NOT rewrite the full reward function.
"""
