import numpy as np
from collections import deque

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
        self.existing_servers = {dc: {gen: deque() for gen in self.server_generations} for dc in self.data_centers}

    def create_action(self, operation, server_generation, server_id, data_center):
        """
        Create a new action tuple.
        """
        return (operation, server_generation, server_id, data_center)

    def generate_server_id(self, server_generation):
        """
        Generate IDs like CPU.S1_1, GPU.S2_3 for FIFO approach.
        :return: A unique server ID string
        """
        self.server_id_counters[server_generation] += 1
        return f"{server_generation}_{self.server_id_counters[server_generation]}"

    def convert_actionspace_to_action(self, action_space):
        """
        Convert the action space (defined as Box) into meaningful actions.
        :param action_space: The action space array from the RL environment.
        :return: The corresponding action tuples.
        """
        actions = []

        for op_index in range(action_space.shape[0]):  # Loop through operation types
            for sg_index in range(action_space.shape[1]):  # Loop through server generations
                for dc_index in range(action_space.shape[2]):  # Loop through data centers
                    if action_space[op_index, sg_index, dc_index] > 0:
                        operation = self.operation_types[op_index]
                        server_generation = self.server_generations[sg_index]
                        data_center = self.data_centers[dc_index]

                        if operation == 'buy':
                            server_id = self.generate_server_id(server_generation)
                            # Add the server to the data center inventory
                            self.existing_servers[data_center][server_generation].append(server_id)
                        
                        elif operation == 'dismiss':
                            if self.existing_servers[data_center][server_generation]:
                                # First-In-First-Out (FIFO) for dismiss
                                server_id = self.existing_servers[data_center][server_generation].popleft()
                            else:
                                continue  # No server to dismiss, skip action

                        elif operation == 'move':
                            if self.existing_servers[data_center][server_generation]:
                                # Last-In-First-Out (LIFO) for move
                                server_id = self.existing_servers[data_center][server_generation].pop()
                            else:
                                continue  # No server to move, skip action
                            # You would typically then move this server to a different data center
                            # This example does not reassign the data center, just demonstrating LIFO

                        elif operation == 'hold':
                            if self.existing_servers[data_center][server_generation]:
                                server_id = self.existing_servers[data_center][server_generation][-1]
                            else:
                                continue  # No server to hold, skip action

                        action = self.create_action(operation, server_generation, server_id, data_center)
                        actions.append(action)

        return actions

# Example usage of convert_actionspace_to_action:
if __name__ == "__main__":
    action_space = ActionSpace()
    
    # Example action space input from the environment
    rl_action_space = np.random.randint(0, 2, size=(4, 7, 4))
    
    actions = action_space.convert_actionspace_to_action(rl_action_space)
    
    for action in actions:
        print(action)