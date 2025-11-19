from src.envs.callback import RenderEveryCallback
from src.envs.env import BattleSnakeEnv
from src.training.ppo import get_ppo_model
from src.envs.wrapper import SafeBattleSnakeEnv

if __name__ == "__main__":
    # Function to create new environments (required by SB3 VecEnv)
    def make_env():
        base_env = BattleSnakeEnv(board_size=11)
        return SafeBattleSnakeEnv(base_env, safe_training_steps=200_000)

    # Create vectorized environments
    model = get_ppo_model(make_env, n_envs=8)

    render_callback = RenderEveryCallback(make_env)

    # Train
    model.learn(total_timesteps=500_000)
    model.save("ppo_battlesnake_safe_3")