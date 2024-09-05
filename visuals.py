import os
import glob
import json
import matplotlib.pyplot as plt

class DatacenterSlotTracker():
    DATACENTERS = ['DC1', 'DC2', 'DC3', 'DC4']
    SERVER_GENERATIONS = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
    SERVER_SLOTS = {
        'CPU.S1': 2, 'CPU.S2': 2, 'CPU.S3': 2, 'CPU.S4': 2,
        'GPU.S1': 4, 'GPU.S2': 4, 'GPU.S3': 4
    }
    MAX_TIMESTEP = 168
    LIFE_EXPECTANCY = 96
    DATACENTER_MAX_SLOTS = {'DC1': 25245, 'DC2': 15300, 'DC3': 7020, 'DC4': 8280}

    def __init__(self, solution_json_dir, json_filename):
        self.json_filename = json_filename
        self.solution_json_dir = solution_json_dir
        self.slots_used = {dc: {gen: [0] * (self.MAX_TIMESTEP + 1) for gen in self.SERVER_GENERATIONS} for dc in self.DATACENTERS}
        self.servers = {}

    def process_actions(self):
        json_file_path = os.path.join(self.solution_json_dir, self.json_filename)
        with open(json_file_path, 'r') as f:
            actions = json.load(f)
            for action in actions:
                dc_id = action['datacenter_id']
                time_step = action['time_step']
                server_gen = action['server_generation']
                server_id = action['server_id']
                slots = self.SERVER_SLOTS[server_gen]
    
                if action['action'] == 'buy':
                    self.servers[server_id] = {
                        'datacenter_id': dc_id,
                        'server_generation': server_gen,
                        'buy_time': time_step
                    }
                    for t in range(time_step, min(time_step + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP)):
                        self.slots_used[dc_id][server_gen][t] += slots
                elif action['action'] == 'dismiss':
                    if server_id in self.servers:
                        buy_time = self.servers[server_id]['buy_time']
                        server_gen = self.servers[server_id]['server_generation']
                        for t in range(time_step, min(buy_time + self.LIFE_EXPECTANCY, self.MAX_TIMESTEP)):
                            self.slots_used[dc_id][server_gen][t] -= slots
                        del self.servers[server_id]

    def plot_results(self):
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        # Reordered for it to look better
        stack_order = ['GPU.S3', 'GPU.S2', 'GPU.S1', 'CPU.S4', 'CPU.S3', 'CPU.S2', 'CPU.S1']

        for i, dc in enumerate(self.DATACENTERS):
            ax = axs[i // 2, i % 2]
            time_steps = range(self.MAX_TIMESTEP + 1)
            data = {gen: [self.slots_used[dc][gen][t] for t in time_steps] for gen in stack_order}
            
            # Calculate the total slots used at each time step
            total_slots = [sum(data[gen][t] for gen in stack_order) for t in time_steps]
            
            ax.stackplot(time_steps, *[data[gen] for gen in stack_order], labels=stack_order)
            # Max available slots line and point of max slots
            max_slots = max(total_slots)
            max_time_step = total_slots.index(max_slots)
            ax.axhline(self.DATACENTER_MAX_SLOTS[dc], color='r', linestyle='--', label=f'Max Slots: {self.DATACENTER_MAX_SLOTS[dc]}')
            ax.annotate(f'Max Slots: {max_slots}', xy=(max_time_step, max_slots), xytext=(max_time_step, max_slots + 10), arrowprops=dict(facecolor='black', shrink=0.05))
            
            ax.set_title(f'Datacenter {dc}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Slots Used')
            ax.legend(loc='upper left')

        plt.tight_layout()
        output_dir = 'visuals'
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.json_filename))[0]
        output_file = os.path.join(output_dir, f"{base_name}_slots_occupancy_by_dc.png")
        plt.savefig(output_file)
        plt.close()
    
    def run(self):
        self.process_actions()
        self.plot_results()

if __name__ == '__main__':
    soluton_dir = 'output/output_gp_dc4_test/'
    json_files = glob.glob(os.path.join(soluton_dir, '*.json'))
    for json_file in json_files:
        tracker = DatacenterSlotTracker(soluton_dir, os.path.basename(json_file))
        tracker.run()
        # break