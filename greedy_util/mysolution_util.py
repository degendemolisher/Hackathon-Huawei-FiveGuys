import numpy as np
import pandas as pd

# setting path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand

from greedy_util import get_solution

seeds = known_seeds('test')

demand = pd.read_csv('data/demand.csv')
for idx, seed in enumerate(seeds):
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    print(f"\nSolution number {idx + 1}.")
    solution = get_solution(actual_demand, 27)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')