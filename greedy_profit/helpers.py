from typing import TypedDict
import numpy as np
import pandas as pd
from utils import load_problem_data
from evaluation import get_capacity_by_server_generation_latency_sensitivity, get_time_step_demand

demand, datacenters, servers, selling_prices = load_problem_data()

# Takes a timestep and returns a list of server types that are available for purcahse 
# at that time step
def get_available_servers(timestep: int) -> tuple[list[str], list[str]]:
    available_servers: list[str] = []

    def check_server_availability(server: np.array):
        availability = eval(server[2])
        if timestep >= availability[0] and timestep <= availability[1]:
            available_servers.append(server[0])

    servers.apply(check_server_availability, axis=1, raw=True)

    available_cpu_servers = [s for s in available_servers if 'CPU' in s]
    available_gpu_servers = [s for s in available_servers if 'GPU' in s]
    return available_cpu_servers, available_gpu_servers

# Dictionary that matches the structure of the actions in solution_example.json
class Action(TypedDict):
    time_step: int
    datacenter_id: str
    server_generation: str
    server_id: str
    action: str

    def columns():
        return ['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action']

# The number of days before the server breaks even (in DC1)
# See the Profitability spreadsheet in the google drive
server_profitability = {
    'CPU.S1': 45,
    'CPU.S2': 35,
    'CPU.S3': 22,
    'CPU.S4': 16,
    'GPU.S1': 13,
    'GPU.S2': 14,
    'GPU.S3': 12
}

def get_most_profitable(servers: list[str]) -> str: 
    return servers[np.argmin([server_profitability[s] for s in servers])]

indexed_servers = servers.set_index('server_generation')
def get_server_capacity(server_generation: str) -> int:
    return indexed_servers.loc[server_generation, 'capacity']

def get_unsatisfied_demand(actual_demand: pd.DataFrame, fleet: list[str], time_step: int):
    current_demand = get_time_step_demand(actual_demand, time_step)
    capacity = get_capacity_by_server_generation_latency_sensitivity(fleet)

    unsatisfied_demand = current_demand - capacity

    return max(0, unsatisfied_demand)

# returns demand based on server generation, server latency and the timestep
def get_server_demand(demand: pd.DataFrame, server_generation: str, datacenter_id: str, timestep: int) -> int:
    server_latency = 'high' if datacenter_id == 'DC3' or datacenter_id == 'DC4' else 'medium' if datacenter_id == 'DC2' else 'low'

    return demand.query(f"time_step == {timestep} and latency_sensitivity == '{server_latency}'")[server_generation]

# this is just to check that all is in check
def print_ser(fleet: pd.DataFrame):
    for index, row in fleet.iterrows():
        print(row.server_generation)