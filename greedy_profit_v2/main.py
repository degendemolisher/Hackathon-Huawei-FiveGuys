import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from greedy_profit_v2.greedy_profit_algorithm import greedy_profit_algorithm
from greedy_profit_v2.results import save_results_as_actions
from seeds import known_seeds
from evaluation import get_actual_demand


def get_solution():
    
    seeds = known_seeds('test')

    demand = pd.read_csv('./data/demand.csv')
    for seed in seeds:
        print('----------------------------')
        print(f'Seed: {seed}')
        # SET THE RANDOM SEED
        np.random.seed(seed)

        # GET THE DEMAND
        actual_demand = get_actual_demand(demand)

        # CALL YOUR APPROACH HERE
        results = greedy_profit_algorithm(actual_demand, 0, float(8))

        # Extract the directory path from the file path
        file_path = f'output/solutions/{seed}.json'
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the results as actions
        save_results_as_actions(file_path, results)

if __name__ == '__main__':
    get_solution()
    print('----------------------------')
    print('All seeds have been processed')