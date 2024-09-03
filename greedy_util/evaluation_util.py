from tqdm import tqdm

import statistics
import sys

# setting path
sys.path.append('..')

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds

best_submit = 351479106.32757604 # Kwun
best_submit = 427370482.7219451  # Artem
best_submit = 402100574.79285 # Artem on test seeds


demand, datacenters, servers, selling_prices = load_problem_data('../data')

solution_scores = []
for seed in tqdm(known_seeds('test')):
    # LOAD SOLUTION
    solution = load_solution(f'output/{seed}.json')
    # solution = load_solution('../data/solution_example.json')

    # EVALUATE THE SOLUTION
    score = evaluation_function(solution,
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                seed=seed,
                                verbose=False)
    
    solution_scores.append(score)
    print(f'Solution score: {score}')

mean = statistics.mean(solution_scores)
increase = round((mean / best_submit) - 1, 1)

print(f'\nMean score: {mean}')
print(f'Increase since the best submit: {increase}')