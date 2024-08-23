import pandas as pd

import system_state

class DecisonMaker:
    def __init__(self, servers: pd.DataFrame, datacenters: pd.DataFrame, demand: pd.DataFrame):
        self.servers = servers
        self.datacenters = datacenters
        self.demand = demand
    
    def decide(self, time_step: int, state: system_state):
        action_dict = {'buy': [], 'move': [], 'dismiss': []}

        # 1. Forecast demand for next time step
        forecasted_demand = self.demand[self.demand['time_step'] == time_step]

        # 2. Calculate required capacity
        required_capacity = self.calculate_required_capacity(forecasted_demand)

        # 3. Server purchase decisions
        for datacenter in self.datacenters.itertuples():
            current_capacity = self.calculate_current_capacity(state, datacenter.datacenter_id)
            if required_capacity[datacenter.datacenter_id] > current_capacity:
                servers_to_buy = self.decide_servers_to_buy(
                    datacenter, 
                    required_capacity[datacenter.datacenter_id] - current_capacity,
                    time_step
                )
                action_dict['buy'].extend(servers_to_buy)

        # 4. Server dismissal decisions
        servers_to_dismiss = self.decide_servers_to_dismiss(state)
        action_dict['dismiss'].extend(servers_to_dismiss)

        # 5. Server movement decisions
        servers_to_move = self.decide_servers_to_move(state, required_capacity)
        action_dict['move'].extend(servers_to_move)

        return action_dict

    def calculate_required_capacity(self, forecasted_demand):
        # Calculate required capacity for each datacenter
        pass

    def calculate_current_capacity(self, state, datacenter_id):
        # Calculate current capacity of a datacenter
        pass

    def decide_servers_to_buy(self, datacenter, capacity_needed, time_step):
        # Decide which servers to buy based on efficiency and availability
        pass

    def decide_servers_to_dismiss(self, state):
        # Decide which servers to dismiss based on age and efficiency
        pass

    def decide_servers_to_move(self, state, required_capacity):
        # Decide which servers to move between datacenters
        pass
