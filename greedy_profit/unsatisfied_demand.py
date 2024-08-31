import numpy as np
import pandas as pd
from ..evaluation import get_capacity_by_server_generation_latency_sensitivity, get_time_step_demand


def get_unsatisfied_demand(actual_demand: pd.DataFrame, fleet: pd.DataFrame, time_step: int):
    """
    Returns a DataFrame of server_generation index and columns for each latency sensitivity.
    Unsatisfied demand is:
    - Positive if there is more demand than the current server capacity can handle
    - Negative if there is more current server capacity than the demand can consume
    """
    current_demand = get_time_step_demand(actual_demand, time_step)
    
    # The fleet is empty, so unsatisfied demand = all demand
    if time_step == 1:
        return current_demand
    
    capacity = get_capacity_by_server_generation_latency_sensitivity(fleet)

    unsatisfied_demand = {}
    relevant_server_generations = np.union1d(current_demand.index.unique(), capacity.index.unique())
    for server_generation in relevant_server_generations:
        unsatisfied_demand[server_generation] = {}
        for latency_sensitivity in np.array(['low', 'medium', 'high']):

            if server_generation in current_demand.index.unique():
                this_demand = current_demand.loc[server_generation][latency_sensitivity]
            else:
                this_demand = 0

            if server_generation in capacity.index.unique():
                this_capacity = capacity.loc[server_generation][latency_sensitivity]
            else:
                this_capacity = 0

            unsatisfied_demand[server_generation][latency_sensitivity] = this_demand - this_capacity
        
    unsatisfied_demand = pd.DataFrame(unsatisfied_demand).transpose().rename_axis('server_generation')
    return unsatisfied_demand