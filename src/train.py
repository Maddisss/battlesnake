
from stable_baselines3.common.vec_env import DummyVecEnv

from env_simulation import SimpleSnakeEnv, preprocess
from ppo import get_ppo_model


if __name__ == "__main__":
    env = DummyVecEnv([lambda: preprocess(SimpleSnakeEnv())])
    model = get_ppo_model(env)
    model.learn(total_timesteps=100000)