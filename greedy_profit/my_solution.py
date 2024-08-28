
import numpy as np
import pandas as pd
from greedy_profit.helpers import *
from seeds import known_seeds
from utils import load_problem_data, save_solution
from evaluation import get_actual_demand, get_known, get_time_step_fleet
from system_state import *

# lifespan threshold as a number of days
LS_THRESHOLD = 50

demand, datacenters, servers, selling_prices = load_problem_data()

action: Action = {
            'action': 'buy',
            'datacenter_id': 'DC1',
            'server_generation': 'CPU.S1',
            'server_id': '1',
            'time_step': '1',
        }

def get_my_solution(actual_demand) -> list[Action]:
    solution = pd.DataFrame(columns=Action.columns())
    system_state = SystemState(datacenters, servers)

    for ts in np.arange(1,get_known('time_steps')):
        fleet = get_time_step_fleet(solution, ts)

        # Checks the demand that needs to be satisfied
        unsatisfied_demand = get_unsatisfied_demand(actual_demand, fleet, ts)

        '''
        maximising lifespan is built into the main algorithm
        you
        for starters, in theory, lifespan is maximised by:
        1. buying servers infrequently (once per generation per release date
            OR buying more servers than currently necessary, expecting demand to rise later)
        2. never overhaul the entire fleet at the same time (resets lifespan to ¬0)
        3. dismissing servers as closely to the end of their lifespan as possible (above a certain threshold)
        4. [HARD] moving servers as much as possible before dismissing them
        '''
        
        # check for low demand servers to dismiss to allow for new servers
        # ideally, servers should be dismissed under these conditions ONLY
        # TODO: update SystemState object with new dismiss actions
        action_list = []

        for index, row in system_state.fleet.iterrows():
            if row.life_span > LS_THRESHOLD:
                row_demand = get_server_demand(demand, row.server_generation, row.datacenter_id, ts)

                if (row.latency_sensitivity == 'high' and row_demand < 50_000) or \
                    (row.latency_sensitivity == 'medium' and row_demand < 100_000) or \
                    (row.latency_sensitivity == 'low' and row_demand < 200_000):
                    Action = {
                            'action': 'dismiss',
                            'datacenter_id': row.datacenter_id,
                            'server_generation': row.server_generation,
                            'server_id': row.server_id,
                            'time_step': 'ts',
                        }

                    action_list.append(Action)

    

        # Buys servers if there is unsatisfied demand
        ## which servers?
        available_cpus, available_gpus = get_available_servers(ts)
        cpu_target = get_most_profitable(available_cpus)
        gpu_target = get_most_profitable(available_gpus)

        ## how many?
        cpu_capacity = get_server_capacity(cpu_target)
        gpu_capacity = get_server_capacity(gpu_target)

        # update solution with new actions at the end of each timestep
        solution = pd.concat([solution, action_list])

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

