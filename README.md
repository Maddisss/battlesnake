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

1. Run Battlesnake locally.

   Start the Agents you want to use:

   Random:

   ```
   python -m src.agents.random_snake 8001
   ```

   Agent:

   ```
   python -m src.agents.ppo_agent_snake 8002
   ```

   Human:

   ```
   python -m src.agents.human_snake 8003
   ```

2. Play a game locally (battlesnake has to be running on port of the agent)

   Single Player:

   ```
   battlesnake play -W 11 -H 11 --name 'Agent' --url http://localhost:<port> -g solo --browser
   ```

   Multiple Players:

   Add `--name "<name>" --url http://localhost:<port>` for each player

   ```
   battlesnake play --width 11 --height 11 --name "Agent" --url http://localhost:8002 --name "Random" --url http://localhost:8001 --browser
   ```

# Eureka

## Install vllm and deploy llm on Docker Open Ai compatible Server

Pull Official vLLM Docker Image

```
docker pull vllm/vllm-openai:latest
```

Run Server Container

```
docker run --gpus all --rm -d \
  -p 8000:8000 \
  --name vllm-server \
  -e CUDA_VISIBLE_DEVICES=0 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype bfloat16 \
  --max-num-batched-tokens 4096 \
  --port 8000
```
