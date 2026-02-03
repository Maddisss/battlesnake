from agents.KILabAgentGroup3.gym_env.battlesnake_env import BattlesnakeEnv

from agents.KILabAgentGroup3.AgentWrapper import AgentWrapper
from environment.Battlesnake.model.Direction import Direction
from environment.battlesnake_environment import BattlesnakeEnvironment

class LLMRewardBattleSnake(BattlesnakeEnv):
    def __init__(self, reward_fn, env: BattlesnakeEnvironment, rl_agent_index=-1, render_mode=None, enemy_classes={}):
        super().__init__(env, rl_agent_index, render_mode, enemy_classes)
        self.reward_fn = reward_fn

    def compute_reward(self, agent: AgentWrapper, action: Direction, terminated: bool, ate_food: bool, died: bool, enemy_died: bool = False) -> tuple[float, dict]:

        try:
            reward, reward_components = self.reward_fn(agent, action, terminated, ate_food, died, enemy_died)
            return float(reward), reward_components
        except Exception as e:
            return -20.0, {}