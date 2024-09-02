import numpy as np
import pandas as pd
from collections import deque, defaultdict
class ActionSpace:
    def __init__(self):
      
        self.operation_types = ['buy', 'move', 'dismiss', 'hold']
        self.server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4',
                                   'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.data_centers = ['DC1', 'DC2', 'DC3', 'DC4']
        self.datacenters_csv = pd.read_csv("data/datacenters.csv")
        self.cpu = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4']
        self.gpu = ['GPU.S1', 'GPU.S2', 'GPU.S3']

        # Track the number of servers bought for each generation
        self.server_id_counters = {gen: 0 for gen in self.server_generations}
        self.existing_servers = {dc: {gen: deque() for gen in self.server_generations} for dc in self.data_centers}
        self.Fleet = {dc: {gen: {'servers': [], 'total_owned': 0} for gen in self.server_generations} for dc in self.data_centers}
        self.servers_acted_upon = set()

    def create_action(self, operation, server_generation, server_id, source_dc, target_dc=None):
        """
        Create a new action tuple. If the action is "move", include both source and target data centers.
        """
        if operation == 'move':
            # including target data center for move actions
            return (operation, server_generation, server_id, source_dc, target_dc)
        else:
            # Return action tuple  
            return (operation, server_generation, server_id, source_dc)
        

    def move_server(self, source_dc, target_dc, server_generation, server_id):
        """
        Move a server from the source data center to the target data center.
        """
        self.existing_servers[source_dc][server_generation].remove(server_id)
        self.existing_servers[target_dc][server_generation].append(server_id)
        
    def generate_server_id(self, server_generation):
        """
        Generate IDs like CPU.S1_1, GPU.S2_3 for FIFO approach.
        :return: A unique server ID string
        """
        self.server_id_counters[server_generation] += 1
        return f"{server_generation}_{self.server_id_counters[server_generation]}"
    def update_fleet(self, data_center, server_generation, server_id, operation, timestep, target_dc=None):
        """
        Update the fleet record when a server is bought, moved, or dismissed.
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
        elif operation == 'move':
            if target_dc:
                # Remove server from source and add to target in the fleet record
                self.Fleet[data_center][server_generation]['servers'] = [
                    s for s in self.Fleet[data_center][server_generation]['servers'] if s['server_id'] != server_id
                ]
                self.Fleet[data_center][server_generation]['total_owned'] -= 1
                self.Fleet[target_dc][server_generation]['servers'].append({'server_id': server_id, 'timestep_moved': timestep})
                self.Fleet[target_dc][server_generation]['total_owned'] += 1
    def reset_action_tracking(self):
        """
        Reset the set that tracks which servers have had actions taken on them for each time step.
        This is called at the beginning of each time step to ensure no carryover of tracked actions.
        """
        self.servers_acted_upon.clear()

    def convert_actionspace_to_action(self, action_space, timestep):
        """
        Convert the action space (defined as Box) into meaningful actions.
        return the corresponding action tuples.
        """
        # Store the actions generated in this time step
        actions = []
        # Reset the tracking set at the beginning of each time step
        self.reset_action_tracking()
        # for operations  
        for op_index in range(action_space.shape[0]):
            # server generations
            for sg_index in range(action_space.shape[1]):
                # data centers
                for dc_index in range(action_space.shape[2]):
                    # Get the quantity for the action from the action space array
                    # each element [op_index, sg_index, dc_index, 0] contains a quantity value.
                    quantity = action_space[op_index, sg_index, dc_index, 0]
                    if quantity > 0:  # Proceed only if there is a positive quantity
                        operation = self.operation_types[op_index]
                        server_generation = self.server_generations[sg_index]
                        data_center = self.data_centers[dc_index]

                        if operation == 'buy':
                            for _ in range(quantity):
                                # Generate a unique server ID
                                server_id = self.generate_server_id(server_generation)
                                if server_id not in self.servers_acted_upon:
                                    # Add the server to the existing servers list
                                    self.existing_servers[data_center][server_generation].append(server_id)
                                    # Create the action and add it to the actions list
                                    action = self.create_action(operation, server_generation, server_id, data_center)
                                    actions.append(action)
                                    # Update the fleet with the new server
                                    self.update_fleet(data_center, server_generation, server_id, operation, timestep)
                                    # Mark the server as acted upon
                                    self.servers_acted_upon.add(server_id)
                        
                        elif operation == 'dismiss':
                            for _ in range(quantity):
                                if self.existing_servers[data_center][server_generation]:
                                    # FIFO (First-In-First-Out) to get the oldest server to dismiss
                                    server_id = self.existing_servers[data_center][server_generation].popleft()
                                    if server_id not in self.servers_acted_upon:
                                        # Create the dismiss action and add it to the actions list
                                        action = self.create_action(operation, server_generation, server_id, data_center)
                                        actions.append(action)
                                        # Update the fleet to reflect the server dismissal
                                        self.update_fleet(data_center, server_generation, server_id, operation, timestep)
                                        # Mark the server as acted upon
                                        self.servers_acted_upon.add(server_id)
                                else:
                                    break  

                        elif operation == 'move':
                            for target_dc_index in range(action_space.shape[2]):
                                if target_dc_index != dc_index:  # Avoid moving to the same DC
                                    target_dc = self.data_centers[target_dc_index]
                                    if self.existing_servers[data_center][server_generation]:
                                        # LIFO (Last-In-First-Out) to get the most recently added server to move
                                        server_id = self.existing_servers[data_center][server_generation].pop()
                                        if server_id not in self.servers_acted_upon:
                                            # Create the move action and add it to the actions list
                                            action = self.create_action(operation, server_generation, server_id, data_center, target_dc)
                                            actions.append(action)
                                            # Move the server between data centers in the existing_servers structure
                                            self.move_server(data_center, target_dc, server_generation, server_id)
                                           
                                            self.update_fleet(data_center, server_generation, server_id, operation, timestep, target_dc)
                                            # Mark the server as acted upon
                                            self.servers_acted_upon.add(server_id)
                                    else:
                                        break  

                        elif operation == 'hold':
                            if self.existing_servers[data_center][server_generation]:
                                # Get the most recently added server for hold operation
                                server_id = self.existing_servers[data_center][server_generation][-1]
                                if server_id not in self.servers_acted_upon:
                                    # Create the hold action and add it to the actions list
                                    action = self.create_action(operation, server_generation, server_id, data_center)
                                    actions.append(action)
                                    # Mark the server as acted upon
                                    self.servers_acted_upon.add(server_id)

        return actions
    
    def convert_actionspace_to_actionV2(self, agent_action):
        actions = []
        for datacenter in range(len(agent_action)):
            datacenter_id = self.data_centers[datacenter]
            #get capacity for the datacenter
            dc_cap = self.datacenters_csv[self.datacenters_csv["datacenter_id"] == datacenter_id]["slots_capacity"].iloc[0]
            #over all server generations
            for server_gen_num in range(len(agent_action[datacenter])):
                server_gen = self.server_generations[server_gen_num]
                for action_perc_num in range(len(agent_action[datacenter][server_gen_num])):
                    action_perc = agent_action[datacenter][server_gen_num][action_perc_num]
                    num_servers = action_perc * dc_cap
                    #divide by slotsize to get number of servers
                    if(server_gen in self.cpu):
                        num_servers /= 2
                    else:
                        num_servers /= 4
                    num_servers = int(num_servers)
                    action = self.operation_types[action_perc_num]
                    actions.append([action, server_gen, num_servers, datacenter_id])
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