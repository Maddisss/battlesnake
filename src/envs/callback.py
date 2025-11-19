from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RenderEveryCallback(BaseCallback):
    def __init__(self, make_env, render_interval=100_000, max_steps=200, verbose=0):
        super().__init__(verbose)
        self.make_env = make_env
        self.render_interval = render_interval
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_interval == 0:
            print(f"\n--- Rendering at {self.num_timesteps} steps ---")
            env = self.make_env()
            obs, _ = env.reset()
            done = False
            total_reward = 0
            for _ in range(self.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                env.render()  # <-- visualize the current board
                if done or truncated:
                    break
            print(f"Episode reward: {total_reward}")
            env.close()
        return True
