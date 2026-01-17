# ppo.py
from stable_baselines3 import PPO
from agents.KILabAgentGroup3.rl_agents.ppo import create_ppo_model

def get_ppo_model(env, flashlight_mode=False):
    # ppo_params = dict(
    #     # Learning dynamics
    #     learning_rate=2e-4,
    #     n_steps = 128, # 1024, # 128 (recurrent)
    #     batch_size= 512, # 512, # 256 (recurrent)
    #     n_epochs=4,

    #     # PPO specifics
    #     gamma=0.995,
    #     gae_lambda=0.95,
    #     clip_range=0.1,
    #     clip_range_vf=0.15,

    #     # Exploration / entropy
    #     ent_coef=0.04,
    #     vf_coef=0.5,

    #     # Stability
    #     max_grad_norm=0.5,

    #     verbose=1,
    #     tensorboard_log="./runs/logs/",
    # )

    # model = create_ppo_model(env=env, flashlight_mode=flashlight_mode, **ppo_params)
    model = load_ppo_model("ppo_fully_obs_stage_1_eureka_600000_steps.zip", env=env)
    return model

def load_ppo_model(path: str, env):
    model = PPO.load(path, env=env)
    return model