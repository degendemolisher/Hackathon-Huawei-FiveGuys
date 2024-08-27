import itertools
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
        self.existing_servers = {dc: deque() for dc in self.data_centers}

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

    def generate_all_possible_actions(self, max_actions=None):
        """
        Generate all possible actions based on the current server inventory.
        
        :param max_actions: Maximum number of actions to generate.
        :return: List of all possible actions
        """
        all_actions = []
        
        for operation in self.operation_types:
            if operation == 'buy':
                for server_gen in self.server_generations:
                    for dc in self.data_centers:
                        action = self.create_action(operation, server_gen, '<new_id>', dc)
                        all_actions.append(action)
                        if max_actions and len(all_actions) >= max_actions:
                            return all_actions

            elif operation == 'move':
                for dc_from in self.data_centers:
                    for server_gen, server_id in list(self.existing_servers[dc_from]):
                        for dc_to in self.data_centers:
                            if dc_from != dc_to:
                                action = self.create_action(operation, server_gen, server_id, dc_to)
                                all_actions.append(action)
                                if max_actions and len(all_actions) >= max_actions:
                                    return all_actions

            elif operation == 'dismiss':
                for dc in self.data_centers:
                    if self.existing_servers[dc]:
                        server_gen, server_id = self.existing_servers[dc].popleft()
                        action = self.create_action(operation, server_gen, server_id, dc)
                        all_actions.append(action)
                        if max_actions and len(all_actions) >= max_actions:
                            return all_actions

            elif operation == 'hold':
                for dc in self.data_centers:
                    for server_gen, server_id in self.existing_servers[dc]:
                        action = self.create_action(operation, server_gen, server_id, dc)
                        all_actions.append(action)
                        if max_actions and len(all_actions) >= max_actions:
                            return all_actions

        return all_actions[:max_actions] if max_actions else all_actions

def test_action_space():
    action_space = ActionSpace()
 
    print("Test 1: max_actions=10")
    actions_10 = action_space.generate_all_possible_actions(max_actions=10)
    print(f"Number of actions generated: {len(actions_10)}")
    for action in actions_10:
        print(action)

    # Test 2: Generate with a larger max_actions limit
    print("\nTest 2: max_actions=50")
    actions_50 = action_space.generate_all_possible_actions(max_actions=50)
    print(f"Number of actions generated: {len(actions_50)}")
    for action in actions_50:
        print(action)

    # Test 3: Check if actions are relevant and within the limit
    assert len(actions_10) <= 10, "Test 1 failed: More than 10 actions generated"
    assert len(actions_50) <= 50, "Test 2 failed: More than 50 actions generated"

    print("\nAll tests passed!")

# Run the test function
test_action_space()

# Example Usage
action_space = ActionSpace()
limited_actions = action_space.generate_all_possible_actions(max_actions=50)

# Print actions to verify
for action in limited_actions:
    print(action)