import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium
from env_simulation import SimpleSnakeEnv


def get_ppo_model(env: gymnasium.Env) -> PPO:
    env = DummyVecEnv([lambda: SimpleSnakeEnv()])
    model = PPO('CnnPolicy', env, verbose=1)
    return model