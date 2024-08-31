from tqdm import tqdm

import statistics
import sys

# setting path
sys.path.append('..')

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds

demand, datacenters, servers, selling_prices = load_problem_data('../data')

solution_scores = []
for seed in tqdm(known_seeds('training')):
    # LOAD SOLUTION
    solution = load_solution(f'output/{seed}.json')

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
    
print(f'\nMean score: {statistics.mean(solution_scores)}')