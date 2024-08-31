import pandas as pd
from greedy_profit.helpers import selling_prices

def get_servers_to_buy(unsatisfied_demand: pd.DataFrame):
    """
    Calcualte how many of each server to buy.
    Returns a DataFrame of index 'server_generation', with columns for each latency_sensitivity
    """
    servers_to_buy = {}
    for server_generation in unsatisfied_demand.index.unique():
        servers_to_buy[server_generation] = {}
        for latency_sensitivty in unsatisfied_demand.columns.unique():
            selling_price = selling_prices\
                .set_index(['server_generation', 'latency_sensitivity'])\
                .loc[server_generation, latency_sensitivty]['selling_price']

            d = unsatisfied_demand.loc[server_generation][latency_sensitivty]

            # TODO: Is math.floor or math.ceil or round() better?
            servers_to_buy[server_generation][latency_sensitivty] = round(d / selling_price)
    
    return pd.DataFrame(servers_to_buy).transpose().rename_axis('server_generation')