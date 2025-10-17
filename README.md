# battlesnake

## Setup

### Prerequisites

1. Install Go https://go.dev/dl/

2. Install battlesnake CLI with go

   ```
   go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
   ```

3. Install UV

   ```
   pip install uv

   ```

### Setup with UV

1. Sync uv env

   ```
   uv sync
   ```

2. activate uv env

   MacOS/Linux:

   ```
    source .venv/bin/activate
   ```

   Windows:

   ```
   .venv\Scripts\Activate.ps1
   ```

## Start Battlesnake

1. Run Battlesnake locally

   ```
   python main.py
   ```

2. Play a game locally (battlesnake has to be running on port 8000)
   ```
   battlesnake play -W 11 -H 11 --name 'Python Starter Project' --url http://localhost:8000 -g solo --browser
   ```
