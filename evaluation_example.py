from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds
import numpy as np

def format_score(score: float) -> str:
    """
    Rounds the score and formats it with commas every 3 digits.
    """
    return "{:,}".format(round(score))

scores = []
seeds = known_seeds()

for i, seed in enumerate(seeds):

    # LOAD SOLUTION
    fleet, pricing_strategy = load_solution(f"./output/solutions/{seed}.json")

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(fleet,
                                pricing_strategy,
                                demand,
                                datacenters,
                                servers,   
                                selling_prices,
                                elasticity,
                                seed=seed,
                                verbose=0)
    print(f"{seed}: {format_score(score)} ({i+1}/{len(seeds)})")
    scores.append(score)

scores = np.array(scores)
mean_score = np.mean(scores)
mean_score = format_score(mean_score)
print(f'Solution score: {mean_score}')