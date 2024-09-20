from tqdm import tqdm

import statistics
import sys
from main import get_solution

# setting path
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (load_problem_data,
                   load_solution)
from evaluation_v6 import evaluation_function
from seeds import known_seeds

best_submit = 351479106.32757604 # Kwun
best_submit = 427370482.7219451  # Artem



demand, datacenters, servers, selling_prices = load_problem_data()

solution_scores = []

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


for seed in tqdm(known_seeds('test')):
    # LOAD SOLUTION
    file_path = f'./output_test/{seed}.json'
    solution = load_solution(file_path)
    
    # EVALUATE THE SOLUTION
    score = evaluation_function(solution,
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                seed=seed,
                                verbose=False)
    
    # Append the solution score to score.txt
    solution_scores.append(score)
    print(f'Solution score for {seed}: {score}')

mean = statistics.mean(solution_scores)
increase = round((mean / best_submit) - 1, 1)

print(f'\nMean score: {mean}')
print(file_path)
print(f'Increase since the best submit: {increase}')