from enum import Enum, auto
from typing import TypedDict
import numpy as np
import numpy.typing as npt
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

def get_unsatisfied_demand(actual_demand, fleet: list[str], timestep: int):
    def get_total_demand():
        # demand = 0
        # for each server in fleet:
            # demand += get the demand from actual_demand
            pass

    demand = get_total_demand()
    capacity = get_capacity_by_server_generation_latency_sensitivity(fleet)

    unsatisfied_demand = demand - capacity

    return max(0, unsatisfied_demand)