import numpy as np
from collections import deque, defaultdict

class ActionSpace:
    def __init__(self):
        self.operation_types = ['buy', 'move', 'dismiss', 'hold']
        self.server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4',
                                   'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.data_centers = ['DC1', 'DC2', 'DC3', 'DC4']
        self.server_id_counters = {gen: 0 for gen in self.server_generations}
        self.existing_servers = {dc: {gen: deque() for gen in self.server_generations} for dc in self.data_centers}
         
        self.Fleet = {dc: {gen: {'servers': [], 'total_owned': 0} for gen in self.server_generations} for dc in self.data_centers}

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

    def update_fleet(self, data_center, server_generation, server_id, operation, timestep):
        """
        Update the fleet record when a server is bought or dismissed.
        """
        if operation == 'buy':
            self.Fleet[data_center][server_generation]['servers'].append({'server_id': server_id, 'timestep_bought': timestep})
            self.Fleet[data_center][server_generation]['total_owned'] += 1
        elif operation == 'dismiss':
            if self.Fleet[data_center][server_generation]['total_owned'] > 0:
                self.Fleet[data_center][server_generation]['servers'] = [
                    s for s in self.Fleet[data_center][server_generation]['servers'] if s['server_id'] != server_id
                ]
                self.Fleet[data_center][server_generation]['total_owned'] -= 1

    def convert_actionspace_to_action(self, action_space, timestep):
        """
        Convert the action space (defined as Box) into meaningful actions.
        :param action_space: The action space array from the RL environment.
        :param timestep: Current timestep in the simulation.
        :return: The corresponding action tuples.
        """
        actions = []

        for op_index in range(action_space.shape[0]):   
            for sg_index in range(action_space.shape[1]):  
                for dc_index in range(action_space.shape[2]):   
                    quantity = action_space[op_index, sg_index, dc_index, 0]
                    if quantity > 0:
                        operation = self.operation_types[op_index]
                        server_generation = self.server_generations[sg_index]
                        data_center = self.data_centers[dc_index]

                        if operation == 'buy':
                            for _ in range(quantity):
                                server_id = self.generate_server_id(server_generation)
                                self.existing_servers[data_center][server_generation].append(server_id)
                                action = self.create_action(operation, server_generation, server_id, data_center)
                                actions.append(action)
                                self.update_fleet(data_center, server_generation, server_id, operation, timestep)
                        
                        elif operation == 'dismiss':
                            for _ in range(quantity):
                                if self.existing_servers[data_center][server_generation]:
                                    # First-In-First-Out (FIFO) for dismiss
                                    server_id = self.existing_servers[data_center][server_generation].popleft()
                                    action = self.create_action(operation, server_generation, server_id, data_center)
                                    actions.append(action)
                                    self.update_fleet(data_center, server_generation, server_id, operation, timestep)
                                else:
                                    break   

                        elif operation == 'move':
                            for _ in range(quantity):
                                if self.existing_servers[data_center][server_generation]:
                                    # Last-In-First-Out (LIFO) for move
                                    server_id = self.existing_servers[data_center][server_generation].pop()
                                    action = self.create_action(operation, server_generation, server_id, data_center)
                                    actions.append(action)
                                else:
                                    break  

                        elif operation == 'hold':
                            if self.existing_servers[data_center][server_generation]:
                                server_id = self.existing_servers[data_center][server_generation][-1]
                                action = self.create_action(operation, server_generation, server_id, data_center)
                                actions.append(action)

        return actions

    def convert_actionspace_to_actionV2(self, agent_action):
        actions = []
        for datacenter in range(len(agent_action)):
            datacenter_id = self.data_centers[datacenter]
            for server_gen in range(len(agent_action[datacenter])):
                action_number = int(agent_action[datacenter,server_gen])
                server_gen = self.server_generations[server_gen]
                action = self.operation_types[action_number]
                actions.append([action, server_gen, "a", datacenter_id])
        return actions
                

if __name__ == "__main__":
    action_space = ActionSpace()
    
    rl_action_space = np.random.randint(0, 10001, size=(4, 7, 4, 1))
    timestep = 1
    
    actions = action_space.convert_actionspace_to_action(rl_action_space, timestep)
    
    for action in actions:
        print(action)
     
    print("\nFleet Status:")
    for dc in action_space.Fleet:
        for sg in action_space.Fleet[dc]:
            print(f"Datacenter: {dc}, Server Generation: {sg}, Fleet: {action_space.Fleet[dc][sg]}")