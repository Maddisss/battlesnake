# ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.training.cnn import SmallCNN

def get_ppo_model(env_factory, n_envs=8):
    env = make_vec_env(env_factory, n_envs=n_envs)

    policy_kwargs = dict(
        normalize_images=False,
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_battlesnake_tb",
    )
    return model

def load_ppo_model(path: str):
    model = PPO.load("ppo_battlesnake_safe")
    return model