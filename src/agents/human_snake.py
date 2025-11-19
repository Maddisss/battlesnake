import keyboard
from flask import Flask, jsonify
import os

app = Flask(__name__)
current_move = "up"

def listen_keys():
    global current_move
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name in ["up", "down", "left", "right"]:
                current_move = event.name
                print(f"Move set to {current_move}")

@app.get("/move")
def move():
    return jsonify({"move": current_move})

@app.get("/info")
def info():
    return {"apiversion": "1", "author": "Human"}

if __name__ == "__main__":
    import threading
    threading.Thread(target=listen_keys, daemon=True).start()
    port = int(os.environ.get("PORT", "8003"))
    app.run(host="0.0.0.0", port=port)
