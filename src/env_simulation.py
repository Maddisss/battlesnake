import numpy as np
import gymnasium
from gymnasium import spaces


class SimpleSnakeEnv(gymnasium.Env):
    def __init__(self, size=11):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(low=0, high=1, shape=(size, size, 3), dtype=np.float32)
        self.reset()

    def reset(self):
        self.snake = [(self.size//2, self.size//2)]
        self.food = self._place_food()
        self.direction = (0, 1)  # moving right
        self.done = False
        return self._get_obs()

    def _place_food(self):
        empty = [(x, y) for x in range(self.size) for y in range(self.size)
                 if (x, y) not in self.snake]
        return empty[np.random.randint(len(empty))]

    def _get_obs(self):
        grid = np.zeros((self.size, self.size, 3), dtype=np.float32)
        for (x, y) in self.snake:
            grid[x, y, 0] = 1.0  # body
        fx, fy = self.food
        grid[fx, fy, 1] = 1.0  # food
        hx, hy = self.snake[0]
        grid[hx, hy, 2] = 1.0  # head
        return grid

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        # Move snake
        x, y = self.snake[0]
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = dirs[action]
        new_head = (x+dx, y+dy)

        # Check collision
        if (not 0 <= new_head[0] < self.size or
            not 0 <= new_head[1] < self.size or
            new_head in self.snake):
            self.done = True
            return self._get_obs(), -1.0, True, {}

        # Move body
        self.snake.insert(0, new_head)
        reward = 0.0
        if new_head == self.food:
            reward = 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self._get_obs(), reward, self.done, {}
    

def preprocess(env):
    # Wrapper to convert int matrix to float32 normalized 0–1
    class Preprocess(gymnasium.ObservationWrapper):
        def observation(self, obs):
            return obs.astype(np.float32) / 6.0  # normalize
    return Preprocess(env)