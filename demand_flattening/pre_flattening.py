import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import load_problem_data
from evaluation import get_actual_demand

DEMAND, DATACENTERS, SERVERS, SELLING_PRICES, PRICE_ELASTICITY = load_problem_data('data')
TEST_SEED = 2381
WINDOW_SIZE = 12

def calculate_moving_average(demand_df, window_size):
    # Group by server_generation
    grouped = demand_df.groupby('server_generation')
    
    # Function to calculate moving average for a group
    def group_moving_average(group):
        for col in ['high', 'low', 'medium']:
            group[col] = group[col].rolling(window=window_size, min_periods=1).mean().round().astype(int)
        return group
    
    # Apply moving average calculation to each group
    return grouped.apply(group_moving_average).reset_index(drop=True)

def ma_demand_plot(actual_demand: pd.DataFrame, window_size: int = 7):
    server_generations = actual_demand['server_generation'].unique()
    
    # Create subplots vertically
    _, axes = plt.subplots(len(server_generations), 1, figsize=(12, 8 * len(server_generations)))
    plt.subplots_adjust(left=0.2)  # Make room for check buttons
    
    colors = {'high': 'red', 'medium': 'green', 'low': 'blue'}
    
    # Plot for each server generation
    for i, generation in enumerate(server_generations):
        df_gen = actual_demand[actual_demand['server_generation'] == generation]
        
        ax = axes[i]
        
        for column in ['high', 'medium', 'low']:
            color = colors[column]
            
            # Plot original data
            ax.plot(df_gen['time_step'], df_gen[column], label=f'{column.capitalize()}', 
                    color=color, marker='o', alpha=0.5)
            
            # Calculate and plot moving average
            moving_avg = df_gen[column].rolling(window=window_size, center=True, min_periods=1).mean()
            ax.plot(df_gen['time_step'], moving_avg, label=f'{column.capitalize()} MA', 
                    color=f'dark{color}', linestyle='-', linewidth=2)
        
        ax.set_title(f'Demand Trajectory for {generation}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Demand')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def test():
    print(f'[TEST MODE]: Seed used: {TEST_SEED}; Window size: {WINDOW_SIZE}')

    np.random.seed(TEST_SEED)

    actual_demand = get_actual_demand(DEMAND)
    
    ma_demand_plot(actual_demand, WINDOW_SIZE)

if __name__ == "__main__":
    test()