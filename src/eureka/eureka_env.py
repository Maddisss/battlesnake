

from src.envs.env import BattleSnakeEnv


class LLMRewardBattleSnake(BattleSnakeEnv):
    def __init__(self, reward_fn):
        super().__init__()
        self.reward_fn = reward_fn

    def compute_reward(self, died, ate_food, killed_enemy, old_dist, new_dist, old_health):
        state = {
            "health": self.health,
            "snake": self.snake,
            "food": self.food,
            "enemy": self.enemy
        }

        next_state = state.copy()
        action = None

        try:
            return float(self.reward_fn(state, action, next_state))
        except Exception:
            return -1.0