
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
    system_state = SystemState(datacenters, servers)

    for ts in np.arange(1,get_known('time_steps')):

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

    

        """
        Greedy profit algorithm (WIP)
        """
        # Checks the demand that needs to be satisfied
        unsatisfied_demand = get_unsatisfied_demand(actual_demand, system_state.fleet, ts)

        # Calcualte how many of each server to buy

        for server_generation in unsatisfied_demand.index.unique():
            for latency_sensitivty in unsatisfied_demand.columns.unique():
                # TODO
                pass
                selling_price = selling_prices\
                    .set_index(['server_generation', 'latency_sensitivity'])\
                    .loc[server_generation, latency_sensitivty]
                
                d = unsatisfied_demand.loc[server_generation][latency_sensitivty]


        # TODO: this is a WIP faster way using DataFrane.map,
        #       it's just difficult to retrieve the column/row values
        # def get_servers_to_buy(demand):
        #     print(demand)
        # servers_to_buy = unsatisfied_demand.map(get_servers_to_buy)

        # update solution with new actions at the end of each timestep
        solution = pd.concat([solution, action_list])


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

