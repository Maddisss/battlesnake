import random
import typing
import sys
import os

from src.rule_based_methods import get_safe_moves
from src.server import run_server

def info() -> typing.Dict:
    return {"apiversion": "1", "author": "RandomBot", "color": "#00FF00", "head": "smile", "tail": "round-bum"}

def start(game_state: typing.Dict):
    print("Random Snake Game Start")

def end(game_state: typing.Dict):
    print("Random Snake Game Over")

def move(game_state: typing.Dict) -> typing.Dict:
    possible_moves = ["up", "down", "left", "right"]
    safe_moves = get_safe_moves(game_state)
    if safe_moves:
        return {"move": random.choice(safe_moves)}
    return {"move": random.choice(possible_moves)}

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    run_server({"info": info, "start": start, "move": move, "end": end}, port=port)
