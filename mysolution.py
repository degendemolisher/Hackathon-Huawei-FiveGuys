
import numpy as np
import pandas as pd
from helpers import Action
from seeds import known_seeds
from utils import load_problem_data, save_solution
from evaluation import get_actual_demand, get_known

demand, datacenters, servers, selling_prices = load_problem_data()


def get_my_solution(actual_demand) -> list[Action]:
    solution: list[Action] = []

    existing_servers = pd.DataFrame(columns=['server_id', 'time_step_bought', 'datacentre_id', 'server_type'])
    
    for t in np.range(1,get_known('time_steps')):
        # Checks the demand that needs to be satisfied
        
        # Buys servers if there is unsatisfied demand
        pass


    return solution


seeds = known_seeds('training')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

