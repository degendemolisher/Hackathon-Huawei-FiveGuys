from typing import TypedDict
import pandas as pd
from utils import load_problem_data

demand, datacenters, servers, selling_prices = load_problem_data()

# Dictionary that matches the structure of the actions in solution_example.json
class Action(TypedDict):
    time_step: int
    datacenter_id: str
    server_generation: str
    server_id: str
    action: str

    def columns():
        return ['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action']

# returns demand based on server generation, server latency and the timestep
def get_server_demand(demand: pd.DataFrame, server_generation: str, datacenter_id: str, timestep: int) -> int:
    server_latency = 'high' if datacenter_id == 'DC3' or datacenter_id == 'DC4' else 'medium' if datacenter_id == 'DC2' else 'low'

    return demand.query(f"time_step == {timestep} and latency_sensitivity == '{server_latency}'")[server_generation]

def dataframe_lookup(dataframe: pd.DataFrame, key_column: str, key_value: any, value_column: str):
    return dataframe.query(f"{key_column} == @key_value")[value_column].iloc[0]