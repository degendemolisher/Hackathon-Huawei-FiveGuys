import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from greedy_profit_v2.greedy_profit_algorithm import greedy_profit_algorithm
from greedy_profit_v2.results import save_results_as_actions
from seeds import known_seeds
from evaluation import get_actual_demand

"""
The entry point for the greedy_profit_v2 approach as outlined in idea.md

Among the files in /greedy_profit_v2/, there are several parameters labelled as "# ADJUSTABLE".
These are values that heavily effect how the algorithm will score and should be adjusted with trial and error for the best results.

Notes regarding the ADJUSTABLEs:
- the merge_threshold is somewhere between x2.5 and x6 of min(length, length_next)
- minimum_range_length is generally optimal at break_even_time * 2. This may be reduced when the move strategy is added

TODO: See idea.md for the steps
- Develop a move strategy to improve lifespan (see idea.md Move Strategy Extension)
    - This is likely to be entirely separate from the main algorithm and will only need to reprocess the final actions

"""

best_config = {
    "3329": {
        "quantile": "40%",
        "range_multiplier": "1"
    },
    "4201": {
        "quantile": "33%",
        "range_multiplier": "8"
    },
    "8761": {
        "quantile": "35%",
        "range_multiplier": "8"
    },
    "2311": {
        "quantile": "37.5%",
        "range_multiplier": "0.8"
    },
    "2663": {
        "quantile": "37.5%",
        "range_multiplier": "0.5"
    },
    "4507": {
        "quantile": "40%",
        "range_multiplier": "2"
    },
    "6247": {
        "quantile": "60%",
        "range_multiplier": "2"
    },
    "2281": {
        "quantile": "0%",
        "range_multiplier": "10"
    },
    "4363": {
        "quantile": "37.5%",
        "range_multiplier": "1"
    },
    "5693": {
        "quantile": "37.5%",
        "range_multiplier": "3"
    }
}

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

        quantile = best_config[str(seed)]['quantile']
        range_multiplier = best_config[str(seed)]['range_multiplier']

        # CALL YOUR APPROACH HERE
        results = greedy_profit_algorithm(actual_demand, float(quantile.strip('%'))/100, float(range_multiplier))

        # Extract the directory path from the file path
        file_path = f'output_test/{seed}.json'
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save the results as actions
        save_results_as_actions(file_path, results)

if __name__ == '__main__':
    get_solution()
    print('----------------------------')
    print('All seeds have been processed')