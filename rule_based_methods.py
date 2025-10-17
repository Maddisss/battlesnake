from collections import deque


def get_safe_moves(game_state: dict) -> list:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    my_body = game_state["you"]["body"]
    my_head = my_body[0]
    my_neck = my_body[1]
    my_tail = game_state["you"]["body"][-1]  # Tail
    my_length = len(my_body)

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

    # Add own body (except tail) to blocked cells
    blocked_cells = set((b["x"], b["y"]) for b in my_body if b != my_tail)

    # Add all other snakes' bodies to blocked cells
    for snake in game_state["board"]["snakes"]:
        for segment in snake["body"]:
            blocked_cells.add((segment["x"], segment["y"]))

    # Mark moves that would hit blocked cells as unsafe
    for move, pos in next_moves.items():
        if (pos["x"], pos["y"]) in blocked_cells:
            is_move_safe[move] = False

    # Only perform flood-fill on moves that are still safe
    for move, pos in next_moves.items():
        if not is_move_safe[move]:
            continue

        region_size = flood_fill(
            start=(pos["x"], pos["y"]),
            blocked=blocked_cells,
            max_limit=my_length * 2,  # snake needs at least this much open space
            board_width=board_width,
            board_height=board_height,
        )

        # If reachable area is too small → trapped region
        if region_size < my_length:
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


def flood_fill(start: tuple, blocked: set, max_limit: int, board_width: int, board_height: int) -> int:
        """
        Efficient flood-fill counting reachable cells.
        Stops early once enough space is found (prunes search).
        """
        q = deque([start])
        visited = set()
        while q:
            cell = q.popleft()
            if cell in visited or cell in blocked:
                continue
            x, y = cell
            if not (0 <= x < board_width and 0 <= y < board_height):
                continue
            visited.add(cell)

            if len(visited) >= max_limit:
                break  # we already have enough space, no need to continue

            q.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        return len(visited)
