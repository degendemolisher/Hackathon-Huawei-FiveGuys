import pandas as pd
import matplotlib.pyplot as plt
import json
import os

class DatacenterVisualizer:
    def __init__(self, datacenter_csv, server_json):
        self.datacenter_csv = datacenter_csv
        self.server_json = server_json
        self.cpu_server_slots = 2
        self.gpu_server_slots = 4
        self.server_life_expectancy = 96

    def read_data(self):
        self.datacenter_df = pd.read_csv(self.datacenter_csv)
        with open(self.server_json, 'r') as file:
            server_data = json.load(file)
        self.server_df = pd.DataFrame(server_data)

    def calculate_slot_occupancy(self):
        # Add a column for the removal time step
        self.server_df['removal_time_step'] = self.server_df['time_step'] + self.server_life_expectancy

        # Create a DataFrame to track the number of slots occupied over time for each datacenter and server generation
        time_steps = range(1, 168)
        datacenters = [f'DC{i}' for i in range(1, 4)]
        server_generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']
        self.slot_counts = pd.DataFrame(0, index=time_steps, columns=pd.MultiIndex.from_product([datacenters, server_generations], names=['datacenter_id', 'server_generation']))

        # Define the slot requirements for each type of server
        slot_requirements = {
            'CPU.S1': 2, 'CPU.S2': 2, 'CPU.S3': 2, 'CPU.S4': 2,
            'GPU.S1': 4, 'GPU.S2': 4, 'GPU.S3': 4
        }

        # Increment the count for each server at its addition time step and decrement at its removal time step
        for _, row in self.server_df.iterrows():
            slots = slot_requirements[row['server_generation']]
            self.slot_counts.loc[row['time_step']:, (row['datacenter_id'], row['server_generation'])] += slots
            self.slot_counts.loc[row['removal_time_step']:, (row['datacenter_id'], row['server_generation'])] -= slots

    def visualize_slots_occupancy_by_dc(self):
        datacenters = self.slot_counts.columns.levels[0]
        _, axes = plt.subplots(nrows=len(datacenters), ncols=1, figsize=(12, 8 * len(datacenters)))

        # Define the desired stack order
        stack_order = ['CPU.S1', 'GPU.S3', 'GPU.S2', 'GPU.S1', 'CPU.S4', 'CPU.S3', 'CPU.S2']

        for ax, datacenter in zip(axes, datacenters):
            datacenter_slot_counts = self.slot_counts[datacenter]

            # Reorder the columns according to the desired stack order
            datacenter_slot_counts = datacenter_slot_counts[stack_order]

            datacenter_slot_counts.plot(kind='area', stacked=True, ax=ax)
            ax.set_title(f'Slots Occupied by Each Type of Server in {datacenter}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Number of Slots Occupied')
            ax.legend(title='Server Generation')
            ax.grid(True)

            # Set x-axis ticks to show fewer timestamps
            tick_interval = 10  # Adjust this value to show fewer/more timestamps
            ax.set_xticks(range(0, len(datacenter_slot_counts), tick_interval))
            ax.set_xticklabels(range(0, len(datacenter_slot_counts), tick_interval))

        # Adjust layout and save the combined plot to a file
        plt.tight_layout()
        output_dir = 'visuals'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'slots_occupied_by_server_type_combined.png')
        plt.savefig(output_file)
        plt.close()

    def run(self):
        self.read_data()
        self.calculate_slot_occupancy()
        self.visualize_slots_occupancy_by_dc()

# Example usage
if __name__ == "__main__":
    datacenter_csv = "data/datacenters.csv"
    server_json = 'greedy_profit_v2/results/861M_gpuall_cpus432_min_length_x2_merge_threshold_x17_seed_123.json'
    visualizer = DatacenterVisualizer(datacenter_csv, server_json)
    visualizer.run()