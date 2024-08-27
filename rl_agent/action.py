import itertools
from collections import defaultdict, deque

class ActionSpace:
    def __init__(self):
        """
        Initialize the ActionSpace with predefined operations, server generations, and data centers.
        """
 
        self.operation_types = ['buy', 'move', 'dismiss', 'hold']
        self.server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4',
                                   'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.data_centers = ['DC1', 'DC2', 'DC3', 'DC4']
        self.server_id_counters = {gen: 0 for gen in self.server_generations}

    def create_action(self, operation, server_generation, server_id, data_center):
        """
        Create a new action tuple.
        """
        return (operation, server_generation, server_id, data_center)

    def generate_server_id(self, server_generation):
        """
        Maake IDs like CPU.S1_1, GPU.S2_3 for FIFO approach
        :return: A unique server ID string
        """
        self.server_id_counters[server_generation] += 1
        return f"{server_generation}_{self.server_id_counters[server_generation]}"

    def generate_all_possible_actions(self):
        """
        Generate all possible actions based on the current server inventory.
        
        :return: List of all possible actions
        """
        all_actions = []
        
        # Generate actions for each operation type
        for operation in self.operation_types:
            if operation == 'buy':
                # For buying, server_id will be generated dynamically, so use a placeholder

                # action = self.create_action(operation, server_gen, '<new_id>', dc)
        
                # For dismiss, use FIFO (first-in-first-out) for server IDs
 
                # For move, use LIFO (last-in-first-out) for server IDs
       
      
                # For other operations, iterate over existing servers

                     return all_actions