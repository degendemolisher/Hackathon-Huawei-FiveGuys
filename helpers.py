from enum import Enum, auto
from typing import TypedDict
import numpy as np
from utils import load_problem_data


demand, datacenters, servers, selling_prices = load_problem_data()

# Takes a timestep and returns a list of server types that are available for purcahse 
# at that time step
def get_available_servers(timestep: int) -> list[str]:
    available_servers: list[str] = []

    def check_server_availability(server: np.array):
        availability = eval(server[2])
        if timestep >= availability[0] and timestep <= availability[1]:
            available_servers.append(server[0])

    servers.apply(check_server_availability, axis=1, raw=True)
    return available_servers


# Dictionary that matches the structure of the actions in solution_example.json
class Action(TypedDict):
    time_step: int
    datacenter_id: str
    server_generation: str
    server_id: str
    action: str
