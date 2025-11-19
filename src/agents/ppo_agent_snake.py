import random
import typing
import sys
import os

import numpy as np
from stable_baselines3 import PPO


from src.training.ppo import load_ppo_model
from src.rule_based_methods import get_safe_moves
from src.server import run_server
from src.utils import game_state_to_obs


model = load_ppo_model('ppo_battlesnake_safe_3')

def info() -> typing.Dict:
    return {"apiversion": "1", "author": "RLBot", "color": "#FF8800", "head": "caffeine", "tail": "bolt"}

def start(game_state: typing.Dict):
    print("RL Snake Start")

def end(game_state: typing.Dict):
    print("RL Snake Over")

def move(game_state: typing.Dict) -> typing.Dict:
    obs = game_state_to_obs(game_state, board_size=game_state["board"]["height"])
    # obs = np.expand_dims(obs, axis=0)
    action, _ = model.predict(obs)
    move_map = {0: "up", 1: "down", 2: "left", 3: "right"}
    chosen_move = move_map[int(action)]

    safe_moves = get_safe_moves(game_state)

    if chosen_move in safe_moves:
        return {"move": chosen_move}
    move = random.choice(safe_moves)
    print("moved randomly: ", move)
    return {"move": move}

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    run_server({"info": info, "start": start, "move": move, "end": end}, port=port)