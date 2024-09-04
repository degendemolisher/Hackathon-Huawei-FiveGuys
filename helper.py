import pandas as pd
import numpy as np

def get_known(attribute):
    """
    Returns a list of known values for a given attribute.
    This is a mock implementation, replace it with the actual logic.
    """
    if attribute == 'latency_sensitivity':
        return ['low', 'medium', 'high']
    elif attribute == 'server_generation':
        return ['G1', 'G2', 'G3', 'G4']
    return []

def get_random_walk(n, min_val, max_val):
    """
    Simulates a random walk. This is a mock implementation, replace with the actual logic.
    """
    return np.random.uniform(min_val, max_val, size=n)

def get_actual_demand(demand):
    actual_demand = []
    for ls in get_known('latency_sensitivity'):
        for sg in get_known('server_generation'):
            d = demand[demand['latency_sensitivity'] == ls]
            sg_demand = d[sg].values.astype(float)
            rw = get_random_walk(sg_demand.shape[0], 0, 2)
            sg_demand += (rw * sg_demand)

            ls_sg_demand = pd.DataFrame()
            ls_sg_demand['time_step'] = d['time_step']
            ls_sg_demand['server_generation'] = sg
            ls_sg_demand['latency_sensitivity'] = ls
            ls_sg_demand['demand'] = sg_demand.astype(int)
            actual_demand.append(ls_sg_demand)

    actual_demand = pd.concat(actual_demand, axis=0, ignore_index=True)
    actual_demand = actual_demand.pivot(index=['time_step', 'server_generation'], columns='latency_sensitivity')
    actual_demand.columns = actual_demand.columns.droplevel(0)
    actual_demand = actual_demand.loc[actual_demand[get_known('latency_sensitivity')].sum(axis=1) > 0]
    actual_demand = actual_demand.reset_index(['time_step', 'server_generation'], col_level=1, inplace=False)
    return actual_demand