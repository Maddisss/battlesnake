from gymnasium import spaces
import gymnasium
import numpy as np


class BattleSnakeEnv(gymnasium.Env):
    def __init__(self, board_size=11):
        super().__init__()
        self.board_size = board_size

        self.observation_space = spaces.Box(low=0, high=5, 
                                            shape=(board_size, board_size), 
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        # Initialize snakes and food here
        return self.board
    
    def step(self, action):
        reward = 0
        done = False
        info = {}
        # Apply action, update board, compute reward
        return self.board, reward, done, info