

from battlesnake.src.eureka.eureka_db import EurekaDB
from battlesnake.src.eureka.generator import load_reward_function
from battlesnake.src.eureka.train_eureka import continue_training


db = EurekaDB()
candidate = dict(db.top_candidates(1)[0])
print("Top Candidate:")

fn = load_reward_function(candidate["code"])

fitness, stats, rewards = continue_training(
    reward_fn=fn,
    model_path='models/eureka_candidate_test.zip',
    timesteps=10.000
)
