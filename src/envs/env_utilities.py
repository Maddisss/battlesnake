

from agents.KILabAgentGroup3.KILabAgent import KILabAgent
from agents.KILabAgentGroup3.gym_env.battlesnake_env import BattlesnakeEnv
from agents.KILabAgentGroup3.rl_agents.ppo import create_multiple_vec_envs, make_env, create_eval_env
from agents.RandomAgent.RandomAgent import RandomAgent
from agents.KILabAgentGroup3.V3AgentFullyObs import V3AgentFullyObs 
from battlesnake.src.eureka.eureka_env import LLMRewardBattleSnake
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.modes.Modes import GameMode
from environment.battlesnake_environment import BattlesnakeEnvironment


def create_env(reward_fn=None, flashlight_mode=False, evaluation_env=False):

    n_envs = 32

    agents_dict = {
        KILabAgent: {"name": "Wilma", "safe_mode": False},
        V3AgentFullyObs: {"model_path": "models/eureka_candidate_g3_1.zip", "name": "V3Agent", "safe_mode": True},
        V3AgentFullyObs: {"model_path": "models/eureka_candidate_g3_1.zip", "name": "V3Agent", "safe_mode": True},
        V3AgentFullyObs: {"model_path": "models/eureka_candidate_g3_1.zip", "name": "V3Agent", "safe_mode": True},
        # V1Agent: {"name": "enemy_v1"},
        # V2Agent: {"model_path": v2_model_path, "name": "WilmaV2", "safe_mode": True},
        # V1Agent: {"name": "V1Enemy"},
    }

    env_kwargs = {
        "width": 15,
        "height": 15,
        "act_timeout": 100,
        "do_render": False,
        "ruleset_settings": RulesetSettings(viewRadius=5 if flashlight_mode else -1),
        "mode": GameMode.STANDARD
    }

    gym_env_kwargs = {
        "rl_agent_index": 0,
        "render_mode": None,
        "reward_fn": reward_fn,
    }
    if evaluation_env:
        gym_env = create_eval_env(
            env_kwargs=env_kwargs,
            agents_dict=agents_dict,
            gym_env_class=LLMRewardBattleSnake,
            gym_kwargs=gym_env_kwargs
        )
    else:
        gym_env = create_multiple_vec_envs(
            gym_env_class=LLMRewardBattleSnake,
            env_kwargs=env_kwargs,
            gym_kwargs=gym_env_kwargs,
            agents_dict=agents_dict,
            n_envs=n_envs
        )

    return gym_env