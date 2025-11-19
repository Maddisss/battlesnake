# orchestrator/duel_launcher.py
import subprocess
import time
import random

def start_snake(path, port):
    return subprocess.Popen(["python", path, str(port)])

def start_duel():
    snakes = [
        ("../random_snake/main.py", 8001),
        ("../rl_snake/main.py", 8002)
    ]
    procs = [start_snake(path, port) for path, port in snakes]
    time.sleep(2)

    cmd = [
        "battlesnake", "play",
        "--name", "Random", "--url", "http://localhost:8001",
        "--name", "RL", "--url", "http://localhost:8002",
        "--width", "11", "--height", "11",
        "--browser"
    ]
    subprocess.run(cmd)

    for p in procs:
        p.terminate()

if __name__ == "__main__":
    start_duel()
