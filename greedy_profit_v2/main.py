if __name__ == '__main__':
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
    results = greedy_profit_algorithm(actual_demand)

    # SAVE YOUR SOLUTION
    save_results_as_actions(f'{seed}.json', results)