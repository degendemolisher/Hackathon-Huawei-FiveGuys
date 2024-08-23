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
        action_history (list): List of all actions taken in the simulation (solution).
    """
    def __init__(self, datacenters):
        self.time_step = 0
        self.fleet = pd.DataFrame(columns=['datacenter_id', 'server_generation', 'server_id', 'action'])
        
        # Track datacenters' capacity
        self.datacenter_capacity = datacenters[['datacenter_id', 'slots_capacity']].copy()
        self.datacenter_capacity['used_slots'] = 0

        self.performance_metrics = {'U': 0, 'L': 0, 'P': 0}
        self.action_history = []

    def apply_complex_action(self, action_dict):
        """
        Applies a complex action to the current state.
        
        action_dict = {
            'buy': [{'datacenter_id': 'DC1', 'server_generation': 'CPU.S1', 'count': 2}, ...],
            'move': [{'from': 'DC1', 'to': 'DC2', 'server_id': 'xyz123', ...}],
            'dismiss': ['server_id1', 'server_id2', ...],
            'hold': ['server_id3', 'server_id4', ...]
        }
        """
        # TODO:
        # Implementation to process each action type
        # Update fleet, datacenter capacities, and log actions
        pass
    
    def update_time(self):
        self.time_step += 1

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

    def log_action(self, datacenter_id, server_generation, server_id, action):
        """
        Logs an action taken in the simulation.

        Args:
            datacenter_id (str): ID of the datacenter where the action is taken.
            server_generation (str): Generation of the server involved in the action.
            server_id (str): ID of the server involved in the action.
            action (str): Type of action taken (buy, move, dismiss).
        """
        action_data = {
            "time_step": self.time_step,
            "datacenter_id": datacenter_id,
            "server_generation": server_generation,
            "server_id": server_id,
            "action": action
        }

        self.action_history.append(action_data)

    def get_formatted_action_history(self):
        return json.dumps(self.action_history)
