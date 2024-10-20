import pandas as pd
from utils import load_problem_data

demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

# Taken manually from the profitability spreadsheet in the google drive
break_even_time_all = {
        'GPU.S3': {
            'low': 12, # DC1
            'medium': 11, # DC2
            'high': 10 # DC3 and DC4
        },
        'GPU.S2': {
            'low': 14, # DC1
            'medium': 13, # DC2
            'high': 12 # DC3 and DC4
        },
        'GPU.S1': {
            'low': 14, # DC1
            'medium': 13, # DC2
            'high': 11 # DC3 and DC4
        },
        'CPU.S4': {
            'low': 16, # DC1
            'medium': 11, # DC2
            'high': 6 # DC3 and DC4
        },
        'CPU.S3': {
            'low': 22, # DC1
            'medium': 14, # DC2
            'high': 8 # DC3 and DC4
        },
        'CPU.S2': {
            'low': 35, # DC1
            'medium': 22, # DC2
            'high': 12 # DC3 and DC4
        },
        'CPU.S1': {
            'low': 45, # DC1
            'medium': 27, # DC2
            'high': 15 # DC3 and DC4
        }
    }

def get_sorted_servers(file_path: str):
    """
    Get list of tuples of (server, latency) pairs from a CSV file.
    The server/latency pairs are in descending order of profitability.
    """
    
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by='profitability', ascending=False)
    sorted_servers = list(df_sorted[['server_generation', 'latency_sensitivity']].itertuples(index=False, name=None))
    return sorted_servers

def get_slot_size(server_generation: str):
    if 'CPU' in server_generation:
        return 2
    else:
        return 4