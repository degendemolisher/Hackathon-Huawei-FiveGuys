
import numpy as np
import pandas as pd
from greedy_profit.helpers import Action, get_available_servers, get_most_profitable, get_server_capacity, get_unsatisfied_demand
from seeds import known_seeds
from utils import load_problem_data, save_solution
from evaluation import get_actual_demand, get_known, get_time_step_fleet

demand, datacenters, servers, selling_prices = load_problem_data()


def get_my_solution(actual_demand) -> list[Action]:
    solution = pd.DataFrame(columns=Action.columns())

    for ts in np.range(1,get_known('time_steps')):
        fleet = get_time_step_fleet(solution, ts)

        # Checks the demand that needs to be satisfied
        unsatisfied_demand = get_unsatisfied_demand(actual_demand, fleet, ts)

        action: Action = {
            'action': 'buy',
            'datacenter_id': 'DC1',
            'server_generation': 'CPU.S1',
            'server_id': '1',
            'time_step': '1',
        }
        
        # Buys servers if there is unsatisfied demand
        ## which servers?
        available_cpus, available_gpus = get_available_servers(ts)
        cpu_target = get_most_profitable(available_cpus)
        gpu_target = get_most_profitable(available_gpus)

        ## how many?
        cpu_capacity = get_server_capacity(cpu_target)
        gpu_capacity = get_server_capacity(gpu_target)


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

