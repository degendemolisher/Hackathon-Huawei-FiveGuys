if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())

import numpy as np
from seeds import known_seeds
from utils import load_problem_data, save_solution
from evaluation import get_actual_demand, get_known
from system_state import *
from greedy_profit.buy_actions import get_buy_actions
from greedy_profit.helpers import *
from greedy_profit.servers_to_buy import get_servers_to_buy
from greedy_profit.unsatisfied_demand import get_unsatisfied_demand

# lifespan threshold as a number of days
LS_THRESHOLD = 50

demand, datacenters, servers, selling_prices = load_problem_data()

def get_my_solution(actual_demand) -> list[Action]:
    system_state = SystemState(datacenters, servers)
    action_list = []

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

        # for index, row in system_state.fleet.iterrows():
        #     if row.life_span > LS_THRESHOLD: # ERROR HERE
        #         row_demand = get_server_demand(demand, row.server_generation, row.datacenter_id, ts) # TODO: this is not the right demand (should be actual demand)

        #         if (row.latency_sensitivity == 'high' and row_demand < 50_000) or \
        #             (row.latency_sensitivity == 'medium' and row_demand < 100_000) or \
        #             (row.latency_sensitivity == 'low' and row_demand < 200_000):
        #             Action = {
        #                     'action': 'dismiss',
        #                     'datacenter_id': row.datacenter_id,
        #                     'server_generation': row.server_generation,
        #                     'server_id': row.server_id,
        #                     'time_step': 'ts',
        #                 }

                    # action_list.append(Action)

    

        """
        Greedy profit algorithm
        """
        # TODO: if there are servers in DC4 and there is space in DC3, move those servers from DC4 to DC3

        unsatisfied_demand = get_unsatisfied_demand(actual_demand, system_state.fleet, ts)
        servers_to_buy = get_servers_to_buy(unsatisfied_demand)
        buy_actions = get_buy_actions(servers_to_buy, system_state.fleet, ts)
        
        print_greedy_profit_time_step_info(buy_actions, ts)
        
        # Add timestep to the actions
        buy_actions_with_time_step = pd.DataFrame(buy_actions).assign(**{"time_step": str(ts)}).to_dict('records')
        action_list.extend(buy_actions_with_time_step)

        try:
            system_state.update_state(buy_actions)
        except Exception as err:
            print(err)
            return action_list

        # print(get_time_step_demand(actual_demand, ts))
        #print_fleet_statistics(system_state.fleet, ts)
        # SystemState.calculate_objective(system_state.fleet, demand, selling_prices, ts, True) #CURRENTLY INACCURATE

    return action_list


seeds = known_seeds('training')
for seed in seeds:
    print(f"------------------------------------")
    print(f"SEED {seed}")

    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./{seed}.json')

