import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from typing import Tuple

import pandas as pd
import numpy as np

from utils import load_problem_data
from evaluation import change_elasticity_format, change_selling_prices_format, get_actual_demand, get_new_demand_for_new_price
from greedy_profit_v2.greedy_profit_algorithm import greedy_profit_algorithm
from greedy_profit_v2.results import save_results_as_actions

DEMAND, DATACENTERS, SERVERS, SELLING_PRICES, PRICE_ELASTICITY = load_problem_data('data')
TEST_SEED = 1097
WINDOW_SIZE = 12


def calculate_moving_average(actual_demand: pd.DataFrame, 
                             window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    moving_average_df = actual_demand.copy()
    
    server_generations = actual_demand['server_generation'].unique()
    
    for generation in server_generations:
        df_gen = moving_average_df[moving_average_df['server_generation'] == generation]
        
        for column in ['high', 'medium', 'low']:
            moving_avg = df_gen[column].rolling(window=window_size, center=True, min_periods=1).mean().round().astype(int)
            moving_average_df.loc[df_gen.index, column] = moving_avg
    
    return moving_average_df


def calculate_new_price(d0, d1, p0, e):
    """
    Calculate the new price based on:
    d0: old demand
    d1: new demand (moving average)
    p0: old price
    e: price elasticity of demand
    """
    # Avoid division by zero
    if d0 == 0:
        return p0
    
    delta_d = (d1 - d0) / d0
    delta_p = delta_d / e
    p1 = p0 * (1 + delta_p)

    # Ensure price is non-negative
    return max(p1, 0)


def adjust_prices_and_demand(actual_demand: pd.DataFrame, 
                             moving_average: pd.DataFrame, 
                             selling_prices: pd.DataFrame, 
                             price_elasticity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    new_prices_rows = []
    new_demand = actual_demand.copy()

    grouped = actual_demand.groupby(['time_step', 'server_generation'])
    ma_grouped = moving_average.groupby(['time_step', 'server_generation'])

    for (time_step, server_gen), group in grouped:
        ma_group = ma_grouped.get_group((time_step, server_gen))
        
        new_prices_row = {'time_step': time_step, 'server_generation': server_gen}
        new_demand_row = {}

        for level in ['high', 'medium', 'low']:
            actual = group[level].values[0]
            target = ma_group[level].values[0]
            current_price = selling_prices.loc[server_gen, level]
            elasticity = price_elasticity.loc[server_gen, level]

            # Calculate new price
            new_price = calculate_new_price(actual, target, current_price, elasticity)
            new_prices_row[level] = new_price

            # Calculate new demand
            new_demand_value = get_new_demand_for_new_price(actual, current_price, new_price, elasticity)
            new_demand_row[level] = new_demand_value

        new_prices_rows.append(new_prices_row)

        # Update new_demand DataFrame
        new_demand.loc[(new_demand['time_step'] == time_step) & 
                       (new_demand['server_generation'] == server_gen), 
                       ['high', 'medium', 'low']] = list(new_demand_row.values())

    # Create new_prices DataFrame from the list of rows
    new_prices = pd.DataFrame(new_prices_rows)

    return new_prices, new_demand


def parse_prices(new_prices: pd.DataFrame) -> dict:
    # Melt the DataFrame to convert columns to rows
    melted = pd.melt(new_prices, 
                     id_vars=['time_step', 'server_generation'], 
                     var_name='latency_sensitivity', 
                     value_name='price')
    
    # Sort the DataFrame
    melted = melted.sort_values(['time_step', 'server_generation', 'latency_sensitivity'])
    pricing_strategy = melted.to_dict('records')
    
    return {"pricing_strategy": pricing_strategy}


def get_prices_and_demand(actual_demand: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    selling_prices = change_selling_prices_format(SELLING_PRICES)
    elasticity = change_elasticity_format(PRICE_ELASTICITY)
    moving_average = calculate_moving_average(actual_demand, WINDOW_SIZE)

    new_prices, new_demand = adjust_prices_and_demand(actual_demand, 
                                                      moving_average, 
                                                      selling_prices, 
                                                      elasticity)
    pricing_dict = parse_prices(new_prices)

    return pricing_dict, new_demand

def main():
    print(f'[TEST MODE]: Seed used: {TEST_SEED}; Window size: {WINDOW_SIZE}')
    
    np.random.seed(TEST_SEED)

    actual_demand = get_actual_demand(DEMAND)
    pricing_dict, new_demand = get_prices_and_demand(actual_demand)

    # GREEDY_PROFIT_V2
    results = greedy_profit_algorithm(new_demand, pricing_dict, 0, float(8))

    # Extract the directory path from the file path
    file_path = f'output/solutions/{TEST_SEED}.json'
    directory = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save the results as actions
    save_results_as_actions(file_path, results)

    # with open('demnad_flattening/pre_flattening_pricing_strategy.json', 'w') as f:
    #     json.dump(prices_dict, f, indent=2)

if __name__ == "__main__":
    main()