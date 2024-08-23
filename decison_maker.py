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
            available_capacity = self.calculate_available_capacity(state, datacenter.datacenter_id)
            if required_capacity[datacenter.datacenter_id] > available_capacity:
                servers_to_buy = self.decide_servers_to_buy(
                    datacenter, 
                    required_capacity[datacenter.datacenter_id] - available_capacity,
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

    def calculate_required_capacity(self, forecasted_demand: pd.DataFrame) -> dict:
        """
        Calculate required capacity for each datacenter, prioritizing datacenters with lower energy costs.
        
        Parameters:
        - forecasted_demand: DataFrame with columns [time_step, latency_sensitivity, CPU.S1, CPU.S2, CPU.S3, CPU.S4, GPU.S1, GPU.S2, GPU.S3]
        
        Returns:
        - dict: A dictionary with datacenter IDs as keys and required capacity as values
        """

        # Main logic:
        # 1. Calculate total demand for each latency sensitivity
        #    Sum up demand for all server generations for each latency sensitivity
        
        # 2. Initialize a dictionary to store required capacity for each datacenter
        
        # 3. Iterate through datacenters:
        #    a. Get the latency sensitivity for the current datacenter
        #    b. Get the total demand for this latency sensitivity
        #    c. Calculate the number of servers needed to meet this demand:
        #       - For each server generation compatible with this latency:
        #         * Calculate how many servers of this type are needed
        #         * Choose the most efficient server type (considering capacity and energy consumption)
        #       - Sum up the slot sizes needed for these servers
        #    d. Check if the calculated capacity exceeds the datacenter's slot capacity:
        #       - If yes, fill to maximum and move excess to next datacenter
        #       - If no, assign the calculated capacity
        
        # 4. After assigning capacities, check if any demand is left unassigned
        #    If yes, try to assign to any datacenter with remaining capacity, 
        #    prioritizing by energy cost
        
        # 5. Return the dictionary of required capacities

        return required_capacities

    def calculate_available_capacity(self, state, datacenter_id):
        # Calculate available capacity of a datacenter
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
