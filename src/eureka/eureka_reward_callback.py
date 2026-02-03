from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from battlesnake.src.eureka.generator import prompt_llm


class EurekaRewardCallback(BaseCallback):
    def __init__(self, window_size=500, eval_every_steps=50000, number_of_envs=32, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.eval_every_steps = eval_every_steps
        self.number_of_envs = number_of_envs

        self.win_buffer = []
        self.component_buffer = []
        self.length_buffer = []

        self.current_components = None
        self.current_lengths = None

        self.win_rate_history = []
        self.reward_component_history = []
        self.llm_summaries = []
        self.batch_stats = []

        # Step counter and trigger flag
        self.step_counter = 0
        self.global_step_counter = 0
        self.wait_for_window_completion = False

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.current_components = [defaultdict(float) for _ in range(n_envs)]
        self.current_lengths = [0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        # Increment step counter
        self.step_counter += 1
        self.global_step_counter += 1

        # Trigger evaluation when eval_every_steps is reached
        if (self.step_counter * self.number_of_envs) >= self.eval_every_steps:
            self.wait_for_window_completion = True
            self.step_counter = 0  # reset for next trigger

        for env_idx in range(len(infos)):
            info = infos[env_idx]
            stats = info.get("stats", {})

            # accumulate reward components
            components = stats.get("reward_components", {})
            for k, v in components.items():
                self.current_components[env_idx][k] += v

            self.current_lengths[env_idx] += 1

            if dones[env_idx]:
                win = stats.get("win", 0)

                self.win_buffer.append(win)
                self.component_buffer.append(dict(self.current_components[env_idx]))
                self.length_buffer.append(self.current_lengths[env_idx])

                # reset env accumulators
                self.current_components[env_idx] = defaultdict(float)
                self.current_lengths[env_idx] = 0

                # Only compute if triggered and window is full
                if len(self.win_buffer) >= self.window_size:
                    if self.wait_for_window_completion:
                        self._compute_window_stats()
                        # reset the trigger until next 50k steps
                        self.wait_for_window_completion = False

                    # reset buffers → non-overlapping windows
                    self.win_buffer = []
                    self.component_buffer = []
                    self.length_buffer = []

        return True

    def _compute_window_stats(self):
        wins = np.array(self.win_buffer)
        win_rate = np.mean(wins)
        avg_length = np.mean(self.length_buffer)

        # Collect all component keys
        all_keys = set()
        for d in self.component_buffer:
            all_keys.update(d.keys())

        means = {}
        stds = {}
        correlations = {}

        for key in all_keys:
            values = np.array([
                d.get(key, 0.0) for d in self.component_buffer
            ])

            means[key] = np.mean(values)
            stds[key] = np.std(values)

            if np.std(values) > 1e-8:
                correlations[key] = np.corrcoef(values, wins)[0, 1]
            else:
                correlations[key] = 0.0

        self.win_rate_history.append(win_rate)
        self.reward_component_history.append(means)

        # summary = self._generate_summary(
        #     win_rate,
        #     avg_length,
        #     means,
        #     stds,
        #     correlations
        # )

        # self.llm_summaries.append(summary)

        self.batch_stats.append({
            "win_rate": float(win_rate),
            "avg_length": float(avg_length),
            "means": means,
            "stds": stds,
            "correlations": correlations,
            "step_recorded": self.global_step_counter * self.number_of_envs
        })

        # if self.verbose:
        #     print(summary)

    def _generate_summary(self, win_rate, avg_length, means, stds, correlations):
        lines = []
        lines.append(f"=== Batch Analysis ({self.window_size} Episodes) ===")
        lines.append(f"Win Rate: {win_rate:.3f}")
        lines.append(f"Average Episode Length: {avg_length:.1f}")
        lines.append("")
        lines.append("Reward Component Statistics:")

        for key in sorted(means.keys()):
            lines.append(
                f"- {key}: "
                f"mean={means[key]:.3f}, "
                f"std={stds[key]:.3f}, "
                f"corr_with_win={correlations[key]:.3f}"
            )

        lines.append("")
        lines.append(
            "Based on the correlations, which reward components "
            "appear beneficial, neutral, or harmful? "
            "Suggest concrete reward weight adjustments."
        )

        prompt = "\n".join(lines)

        llm_response = prompt_llm(prompt)

        return f"{prompt}\n\nLLM Recommendation:\n{llm_response}"
