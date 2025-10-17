

def get_safe_moves(game_state: dict) -> list:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    my_body = game_state["you"]["body"]
    my_head = my_body[0]
    my_neck = my_body[1]
    print(game_state)

    blocked_cells = set()
    next_moves = {
        "up":    {"x": my_head["x"], "y": my_head["y"] + 1},
        "down":  {"x": my_head["x"], "y": my_head["y"] - 1},
        "left":  {"x": my_head["x"] - 1, "y": my_head["y"]},
        "right": {"x": my_head["x"] + 1, "y": my_head["y"]},
    }

    # Prevent moving backwards
    if my_neck["x"] < my_head["x"]:
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:
        is_move_safe["up"] = False

    # Prevent moving out of bounds
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False


    # Add all snakes' bodies to blocked cells
    for snake in game_state["board"]["snakes"]:
        tail = snake["body"][-1]
        for segment in snake["body"]:
            if not segment == tail:  # exclude tail
                blocked_cells.add((segment["x"], segment["y"]))

    # Mark moves that would hit blocked cells as unsafe
    for move, pos in next_moves.items():
        if (pos["x"], pos["y"]) in blocked_cells:
            is_move_safe[move] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    return safe_moves

