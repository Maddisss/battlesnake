import numpy as np

def game_state_to_obs(game_state, board_size=11):
    """
    Convert Battlesnake API game_state into the (3, board_size, board_size)
    observation expected by the PPO model.
    """
    obs = np.zeros((3, board_size, board_size), dtype=np.uint8)

    you = game_state["you"]
    board = game_state["board"]

    # --- Snake body ---
    for body_part in you["body"][1:]:
        x, y = body_part["x"], body_part["y"]
        obs[0, y, x] = 255  # channel 0

    # --- Snake head ---
    head = you["head"]
    obs[1, head["y"], head["x"]] = 255  # channel 1

    # --- Food ---
    for f in board["food"]:
        obs[2, f["y"], f["x"]] = 255  # channel 2

    return obs
