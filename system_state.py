import json
import pandas as pd


class SystemState:
    """
    Represents the current state of the data center system.

    This class manages the state of the data centers, including server fleet,
    datacenter capacities, performance metrics, and action history.

    Attributes:
        time_step (int): The current time step in the simulation.
        fleet (pd.DataFrame): DataFrame containing information about all servers in the fleet.
        datacenter_capacity (pd.DataFrame): DataFrame tracking datacenters' slot capacities.
        performance_metrics (dict): Dictionary storing current U, L, and P metrics.
        solution (list): List of all actions taken in the simulation.
    """


    def __init__(self, datacenters: pd.DataFrame, servers: pd.DataFrame):
        self.time_step = 1
        self.fleet = pd.DataFrame(columns=[
            'datacenter_id', 
            'server_generation', 
            'server_id',
            # 'time_step_of_purchase',
            'lifespan' # UGLY CODE WARNING
            # 'latency_sensitivity'
        ])
        
        # Track datacenters' slot capacity
        self.datacenter_capacity = datacenters[['datacenter_id', 'slots_capacity']].copy()
        self.datacenter_capacity['used_slots'] = 0

        self.performance_metrics = {'U': 0, 'L': 0, 'P': 0}
        self.solution = []

        self.servers_info = servers
        self.datacenter_info = datacenters


    def update_state(self, decision):
        """
        Update the system state based on a given decision.

        Warning:
            TIME SHOULD BE UPDATED FIRST!
        """
        self.update_time()
        self.update_solution(decision)
        self.update_fleet(decision)
        self.update_datacenter_capacity()


    def update_solution(self, decisions):
        """
        Updates the solution list with new decisions made
        
        Args:
            decisions (list of dict): A list of decision dictionaries. Each dictionary
                should contain the following keys:
                - 'action': str, one of 'buy', 'dismiss', or 'move'
                - 'datacenter_id': str, the ID of the datacenter involved
                - 'server_generation': str, the generation of the server
                - 'server_id': str, the unique ID of the server

        Example:    
            decisins = [
                {
                    'datacenter_id': 'DC1',
                    'server_generation': 'CPU.S1',
                    'server_id': '7f6edd8e-a815-4d5f-ba30-05caf6f90696',
                    'action': 'buy'
                },
                ...
            ]
        """
        for decision in decisions:
            self.solution.append({
                "time_step": self.time_step,
                "datacenter_id": decision["datacenter_id"],
                "server_generation": decision["server_generation"],
                "server_id": decision["server_id"],
                "action": decision["action"]
            })
    

    def update_fleet(self, decisions):
        """
        Update the server fleet based on the given decisions.

        Args:
            decisions (list of dict): A list of decision dictionaries. Each dictionary
                should contain the following keys:
                - 'action': str, one of 'buy', 'dismiss', or 'move'
                - 'datacenter_id': str, the ID of the datacenter involved
                - 'server_generation': str, the generation of the server
                - 'server_id': str, the unique ID of the server
        """
        for decision in decisions:
            if decision['action'] == 'buy':
                latency_sensitivity = self.datacenter_info.loc[
                    self.datacenter_info['datacenter_id'] == decision['datacenter_id'], 
                    'latency_sensitivity'
                ].iloc[0]

                new_server = pd.DataFrame({
                    'datacenter_id': [decision['datacenter_id']],
                    'server_generation': [decision['server_generation']],
                    'server_id': [decision['server_id']],
                    # 'time_step_of_purchase': [self.time_step],
                    'lifespan': [0] # UGLY CODE WARNING
                    # 'latency_sensitivity': [latency_sensitivity]
                })

                self.fleet = pd.concat([self.fleet, new_server], ignore_index=True)

            elif decision['action'] == 'dismiss':
                self.fleet = self.fleet[self.fleet['server_id'] != decision['server_id']]

            elif decision['action'] == 'move':
                self.fleet.loc[self.fleet['server_id'] == decision['server_id'], 'datacenter_id'] = decision['datacenter_id']


    def update_datacenter_capacity(self):
        """
        Update the used slot capacity for each datacenter based on the current fleet.

        Raises:
            ValueError: If any datacenter's used slots exceed its total capacity.

        Returns:
            None
        """
        # Group servers by datacenter and server generation
        server_counts = (
            self.fleet
            .groupby(['datacenter_id', 'server_generation'])
            .size()
            .reset_index(name='count')
        )

        # Merge with server info to get slot sizes
        server_counts = server_counts.merge(self.servers_info[['server_generation', 'slots_size']], on='server_generation')
        
        # Calculate total slots used for each datacenter
        datacenter_slots = (
            server_counts
            .groupby('datacenter_id')
            .apply(lambda x: (x['count'] * x['slots_size']).sum())
            .reset_index(name='used_slots')
        )
        
        # Update used slots in datacenter_capacity
        for _, row in datacenter_slots.iterrows():
            dc_capacity = self.datacenter_capacity.loc[
                self.datacenter_capacity['datacenter_id'] == row['datacenter_id'], 
                'slots_capacity'
            ].iloc[0]  # Get the single value
            
            # Ensure capacity constraints are not violated
            if row['used_slots'] <= dc_capacity:
                self.datacenter_capacity.loc[
                    self.datacenter_capacity['datacenter_id'] == row['datacenter_id'], 
                    'used_slots'
                ] = row['used_slots']
            else:
                raise ValueError(f"Datacenter '{row['datacenter_id']}': capacity exceeded")

    # UGLY CODE WARNING
    def update_time(self):
        self.time_step += 1

        for _, server in self.fleet.iterrows():
            server['lifespan'] += 1


    def update_metrics(self, **kwargs):
        """
        Updates the performance metrics.

        Args:
            **kwargs: Keyword arguments for metrics to update (U, L, P).

        Raises:
            ValueError: If no valid metrics are provided or if an invalid metric is given.
        """
        if not kwargs:
            raise ValueError("At least one metric (U, L, or P) must be provided for update.")
        
        for metric, value in kwargs.items():
            match metric:
                case 'U':
                    self.performance_metrics['U'] = value
                case 'L':
                    self.performance_metrics['L'] = value
                case 'P':
                    self.performance_metrics['P'] = value
                case _:
                    raise ValueError(f"Invalid metric: {metric}. Valid metrics are U, L, and P.")

    # I WANT TO CRY
    # def get_server_ages(self) -> pd.Series:
    #     return self.fleet['time_step_of_purchase'].apply(lambda x: self.time_step - x)


    def get_formatted_solution(self):
        return json.dumps(self.solution)
