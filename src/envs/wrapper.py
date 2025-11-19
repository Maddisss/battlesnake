import numpy as np
import random
from gymnasium import Wrapper

from src.envs.env import BattleSnakeEnv
from src.rule_based_methods import get_safe_moves



class SafeBattleSnakeEnv(Wrapper):
    """
    Gym wrapper to enforce early safe-move training.
    After `safe_training_steps`, all moves are allowed.
    """

    def __init__(self, env: BattleSnakeEnv, safe_training_steps=1e5):
        super().__init__(env)
        self.safe_training_steps = safe_training_steps
        self.total_steps = 0
        self.use_safe_moves = True

    def step(self, action):
        self.total_steps += 1
        if self.total_steps > self.safe_training_steps:
            self.use_safe_moves = False

        # If safe moves are active, restrict actions
        if self.use_safe_moves:
            safe_moves_indices = self._get_safe_actions()
            if action not in safe_moves_indices:
                action = random.choice(safe_moves_indices)

        return self.env.step(action)

    def _get_safe_actions(self):
        # Convert current environment state to dummy game_state for get_safe_moves
        snake_body = self.env.snake
        game_state = {
            "turn": self.total_steps,
            "board": {
                "width": self.env.board_size,
                "height": self.env.board_size,
                "snakes": [{"body": [{"x": c, "y": r} for (r, c) in snake_body]}],
            },
            "you": {"body": [{"x": c, "y": r} for (r, c) in snake_body]},
        }

        safe_moves = get_safe_moves(game_state)
        moves = ["up", "down", "left", "right"]
        safe_indices = [moves.index(m) for m in safe_moves if m in moves]
        if not safe_indices:
            return list(range(4))  # fallback to all actions
        return safe_indices
