
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution, load_problem_data
from evaluation import get_actual_demand

from solution_artem_v1 import get_solution


def get_my_solution(d):
    # This is just a placeholder.
    _, datacenters, servers, _ = load_problem_data()
    return get_solution(datacenters, servers, d)


seeds = known_seeds('training')

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution()

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

