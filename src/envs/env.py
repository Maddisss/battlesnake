import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from src.rule_based_methods import get_safe_moves


class BattleSnakeEnv(gym.Env):
    """
    Battlesnake environment with explicit reward function emphasizing combat and adaptive food value.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=11, max_steps=300, max_health=100):
        super().__init__()
        self.board_size = board_size
        self.max_steps = max_steps
        self.max_health = max_health

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, board_size, board_size), dtype=np.uint8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.done = False

        # Player + Enemy initialization
        self.snake = [(self.board_size // 2, self.board_size // 2),
                      (self.board_size // 2, self.board_size // 2 - 1)]
        self.enemy = [(1, 1), (1, 0)]

        # Hunger system
        self.health = self.max_health

        self._place_food()
        self.prev_distance_to_food = self._food_distance()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # --- MOVE ENEMY FIRST ---
        enemy_action = None
        if self.enemy:
            enemy_action = self._get_enemy_safe_action()
            self._move_snake(self.enemy, enemy_action)

        # --- MOVE PLAYER ---
        terminated = False
        reward = 0.0
        terminated |= self._check_collision(self.snake, action)

        if not terminated:
            self._move_snake(self.snake, action)

        # Check food consumption
        ate_food = False
        if self.snake[0] == self.food:
            ate_food = True
            self._place_food()
        else:
            self.snake.pop()  # normal move

        # Check enemy collision (kills enemy if it ran into you)
        enemy_eliminated = False
        if self.enemy:
            enemy_eliminated = self._check_enemy_collision()

        # Compute reward
        reward = self.compute_reward(terminated, ate_food, enemy_eliminated,
                                    self.prev_distance_to_food, self._food_distance(),
                                    self.health)
        self.prev_distance_to_food = self._food_distance()

        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}


    # ------------------------------------------------------------------
    # 🔹 Explicit reward function
    # ------------------------------------------------------------------
    def compute_reward(self, died, ate_food, killed_enemy, old_dist, new_dist, old_health):
        reward = 0.0
        breakdown = {}

        # Kill enemy
        kill_reward = 10.0 if killed_enemy else 0.0
        reward += kill_reward
        breakdown['killed_enemy'] = kill_reward

        # Death penalty
        death_penalty = -10.0 if died else 0.0
        reward += death_penalty
        breakdown['died'] = death_penalty

        # Food reward scaled by hunger
        if ate_food:
            hunger_factor = (1.0 - old_health / self.max_health)
            food_reward = 1.0 + 4.0 * hunger_factor
        else:
            food_reward = 0.0
        reward += food_reward
        breakdown['ate_food'] = food_reward

        # Small survival reward
        survival_reward = 0.01 if not died else 0.0
        reward += survival_reward
        breakdown['survival'] = survival_reward

        # --- Discourage self-trapping ---
        reachable_area = self._compute_reachable_area(self.snake[0])
        if reachable_area > 0 and reachable_area < 6:
            trap_penalty = -((1/reachable_area)/10)
        elif reachable_area == 0:
            trap_penalty = -0.1
        else:
            trap_penalty = 0
        reward += trap_penalty
        breakdown['self_trap'] = trap_penalty

        # --- Discourage risky head-to-head collisions ---
        if self.enemy:
            enemy_head = self.enemy[0]
            sr, sc = self.snake[0]
            hr, hc = enemy_head
            # Manhattan distance
            head_distance = abs(sr - hr) + abs(sc - hc)
            if head_distance == 1:  # adjacent to enemy head
                head_collision_penalty = -1
            else:
                head_collision_penalty = 0.0
        else:
            head_collision_penalty = 0.0
        reward += head_collision_penalty
        breakdown['head_collision_risk'] = head_collision_penalty

        print(f"Reward breakdown: {breakdown} | Total reward: {reward:.3f}")
        return reward


    # ------------------------------------------------------------------
    # 🔹 Combat rules
    # ------------------------------------------------------------------
    def _check_enemy_collision(self):
        """
        Returns True if enemy is eliminated (enemy head collides with player's body).
        """
        if not self.enemy:
            return False

        enemy_head = self.enemy[0]

        # Enemy head runs into player's body (not just head)
        if enemy_head in self.snake:
            self.enemy = []  # enemy eliminated
            return True

        # Optional: If you also want player head into enemy body as kill
        # if self.snake[0] in self.enemy:
        #     self.enemy = []
        #     return True

        return False


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _move_snake(self, snake, action):
        head_r, head_c = snake[0]
        if action == 0:  # up
            head_r += 1
        elif action == 1:  # down
            head_r -= 1
        elif action == 2:  # left
            head_c -= 1
        elif action == 3:  # right
            head_c += 1
        snake.insert(0, (head_r, head_c))

    def _check_collision(self, snake, action):
        head_r, head_c = snake[0]
        if action == 0:
            head_r += 1
        elif action == 1:
            head_r -= 1
        elif action == 2:
            head_c -= 1
        elif action == 3:
            head_c += 1
        new_head = (head_r, head_c)

        # Walls
        if not (0 <= head_r < self.board_size and 0 <= head_c < self.board_size):
            return True
        # Self collision
        if new_head in snake:
            return True
        return False

    def _place_food(self):
        occupied = set(self.snake + self.enemy)
        empty = [(r, c) for r in range(self.board_size)
                 for c in range(self.board_size)
                 if (r, c) not in occupied]
        self.food = random.choice(empty) if empty else (0, 0)

    def _food_distance(self):
        sr, sc = self.snake[0]
        fr, fc = self.food
        return np.sqrt((sr - fr) ** 2 + (sc - fc) ** 2)

    def _get_obs(self):
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.uint8)
        for r, c in self.snake[1:]:
            obs[0, r, c] = 255
        hr, hc = self.snake[0]
        obs[1, hr, hc] = 255
        fr, fc = self.food
        obs[2, fr, fc] = 255
        return obs

    def _get_enemy_safe_action(self):
        if not self.enemy:
            return None  # Enemy is dead, no action needed
        snake_body = self.enemy
        game_state = {
            "turn": self.steps,
            "board": {
                "width": self.board_size,
                "height": self.board_size,
                "snakes": [{"body": [{"x": c, "y": r} for (r, c) in self.enemy]}],
            },
            "you": {"body": [{"x": c, "y": r} for (r, c) in self.enemy]},
        }

        safe_moves = get_safe_moves(game_state)
        if isinstance(safe_moves, dict) or not safe_moves:
            safe_moves = ["up", "down", "left", "right"]

        moves = ["up", "down", "left", "right"]
        safe_indices = [moves.index(m) for m in safe_moves if m in moves]
        if not safe_indices:
            safe_indices = list(range(4))
        return random.choice(safe_indices)

    def _compute_reachable_area(self, start_pos):
        """
        Flood-fill from start_pos to count empty cells reachable without colliding with self or walls.
        """
        from collections import deque

        visited = set()
        queue = deque([start_pos])
        # Exclude head from occupied cells
        occupied = set(self.snake[1:] + self.enemy)

        count = 0
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or (r, c) in occupied:
                continue
            visited.add((r, c))
            count += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    queue.append((nr, nc))
        return count
