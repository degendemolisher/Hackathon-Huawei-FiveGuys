import ast
import json
import pandas as pd

from evaluation import *


class SystemState:
    """
    Represents the current state of the data center system.

    This class manages the state of the data centers, including server fleet,
    datacenter capacities, performance metrics, and action history.

    Attributes:
        time_step (int): The current time step in the simulation.
        fleet (pd.DataFrame): DataFrame containing information about all servers in the fleet.
        datacenter_capacity (pd.DataFrame): DataFrame tracking datacenters' slot capacities.
        performance_metrics (dict): Dictionary storing current U, L, and P metrics.
        solution (list): List of all actions taken in the simulation.
    """


    def __init__(self, datacenters: pd.DataFrame, servers: pd.DataFrame):
        self.time_step = 0
        self.fleet = pd.DataFrame({
            'datacenter_id':           pd.Series(dtype='string'),
            'server_generation':       pd.Series(dtype='string'),
            'server_id':               pd.Series(dtype='string'),
            'lifespan':                pd.Series(dtype='int'),
            'life_expectancy':         pd.Series(dtype='int'),
            'latency_sensitivity':     pd.Series(dtype='string'),
            'capacity':                pd.Series(dtype='int'),
            'purchase_price':          pd.Series(dtype='int'),
            'average_maintenance_fee': pd.Series(dtype='int'),
            'energy_consumption':      pd.Series(dtype='int'),
            'cost_of_energy':          pd.Series(dtype='int'),
            'moved':                   pd.Series(dtype='int')
        })
        
        # Track datacenters' slot capacity
        self.datacenter_capacity = datacenters[['datacenter_id', 'slots_capacity']].copy()
        self.datacenter_capacity['used_slots'] = 0
        self.datacenter_capacity['total_capacity'] = 0

        self.performance_metrics = {'U': 0, 'L': 0, 'P': 0}
        self.solution = []

        # Convert the 'release_time' column from string to list
        servers['release_time'] = servers['release_time'].apply(ast.literal_eval)
        self.servers_info = servers
        self.datacenter_info = datacenters


    def update_state(self, decision):
        """
        Update the system state based on a given decision.

        Warning:
            TIME SHOULD BE UPDATED FIRST!
        """
        self.update_time()
        self.update_solution(decision)
        self.update_fleet(decision)
        self.update_datacenter_capacity()


    def update_solution(self, decisions):
        """
        Updates the solution list with new decisions made
        
        Args:
            decisions (list of dict): A list of decision dictionaries. Each dictionary
                should contain the following keys:
                - 'action': str, one of 'buy', 'dismiss', or 'move'
                - 'datacenter_id': str, the ID of the datacenter involved
                - 'server_generation': str, the generation of the server
                - 'server_id': str, the unique ID of the server

        Example:    
            decisins = [
                {
                    'datacenter_id': 'DC1',
                    'server_generation': 'CPU.S1',
                    'server_id': '7f6edd8e-a815-4d5f-ba30-05caf6f90696',
                    'action': 'buy'
                },
                ...
            ]
        """
        for decision in decisions:
            self.solution.append({
                'time_step': self.time_step,
                **decision
            })
    

    def update_fleet(self, decisions):
        """
        Update the server fleet based on the given decisions.
        
        Args:
            decisions (list of dict): A list of decision dictionaries. Each dictionary
                should contain the following keys:
                - 'action': str, one of 'buy', 'dismiss', or 'move'
                - 'datacenter_id': str, the ID of the datacenter involved
                - 'server_generation': str, the generation of the server
                - 'server_id': str, the unique ID of the server
        """
        buy_decisions = []
        dismiss_server_ids = []
        move_decisions = []

        for decision in decisions:
            if decision['action'] == 'buy':
                buy_decisions.append(decision)
            elif decision['action'] == 'dismiss':
                dismiss_server_ids.append(decision['server_id'])
            elif decision['action'] == 'move':
                move_decisions.append(decision)

        if buy_decisions:
            buy_df = pd.DataFrame(buy_decisions)
            
            # print(buy_df['datacenter_id'].dtype)
            # print(buy_df['datacenter_id'].head())
            # print(self.datacenter_info['datacenter_id'].dtype)
            # print(self.datacenter_info['datacenter_id'].head())

            # # Check for any list-like values
            # print(buy_df['datacenter_id'].apply(lambda x: isinstance(x, list)).any())
            # print(self.datacenter_info['datacenter_id'].apply(lambda x: isinstance(x, list)).any())
            
            # Merge with datacenter_info
            buy_df = buy_df.merge(
                self.datacenter_info, 
                on='datacenter_id', how='left'
            )
            
            # Merge with servers_info
            buy_df = buy_df.merge(
                # self.servers_info,
                self.servers_info.drop('release_time', axis=1), 
                on='server_generation', how='left'
            )
            # 3/4 columns from datacenetrs.csv
            # 7/10 columns from servers.csv
            # Create DataFrame for new servers
            new_servers = pd.DataFrame({
                'datacenter_id':           buy_df['datacenter_id'],
                'server_generation':       buy_df['server_generation'],
                'server_id':               buy_df['server_id'],
                'lifespan':                0,
                'life_expectancy':         buy_df['life_expectancy'],
                'latency_sensitivity':     buy_df['latency_sensitivity'],
                'capacity':                buy_df['capacity'],
                'purchase_price':          buy_df['purchase_price'],
                'average_maintenance_fee': buy_df['average_maintenance_fee'],
                'energy_consumption':      buy_df['energy_consumption'],
                'cost_of_energy':          buy_df['cost_of_energy'],
                'moved':                   0
            })

            self.fleet = pd.concat([self.fleet, new_servers], ignore_index=True)


        if dismiss_server_ids:
            self.fleet = self.fleet[~self.fleet['server_id'].isin(dismiss_server_ids)]

        if move_decisions:
            # Create mapping of server_id to new datacenter_id
            move_dict = { d['server_id']: d['datacenter_id'] for d in move_decisions }

            # Servers to be moved
            move_mask = self.fleet['server_id'].isin(move_dict.keys())

            # Update datacenter_id for moved servers
            self.fleet.loc[move_mask, 'datacenter_id'] = self.fleet.loc[move_mask, 'server_id'].map(move_dict)
            self.fleet.loc[move_mask, 'moved'] += 1


    def update_datacenter_capacity(self):
        """
        Update the used slot capacity and total capacity for each datacenter based on the current fleet.

        Raises:
            ValueError: If any datacenter's used slots exceed its total capacity.

        Returns:
            None
        """
        if self.fleet.empty:
            return
        
        # Group servers by datacenter and server generation
        server_counts = (
            self.fleet
            .groupby(['datacenter_id', 'server_generation'])
            .size()
            .reset_index(name='count')
        )

        # Merge with server info to get slot sizes
        server_counts = server_counts.merge(
            self.servers_info[['server_generation', 'slots_size', 'capacity']], 
            on='server_generation'
        )
        
        # Calculate total slots used and total capacity for each datacenter
        datacenter_stats = (
            server_counts
            .groupby('datacenter_id')
            .apply(lambda x: pd.Series({
                'used_slots': (x['count'] * x['slots_size']).sum(),
                'total_capacity': (x['count'] * x['capacity']).sum()
            }))
            .rename(columns={'count': 'used_slots', 'capacity': 'total_capacity'})
            .reset_index()
        )

        # Calculate total slots used and total capacity for each datacenter
        # datacenter_stats = (
        #     server_counts
        #     .groupby('datacenter_id')
        #     .agg({
        #         'count': lambda x: (x * server_counts.loc[x.index, 'slots_size']).sum(),
        #         'capacity': lambda x: (x * server_counts.loc[x.index, 'capacity']).sum()
        #     })
        #     .rename(columns={'count': 'used_slots', 'capacity': 'total_capacity'})
        #     .reset_index()
        # )

        # Ensure all datacenters are present in the stats
        all_datacenters = self.datacenter_capacity['datacenter_id'].unique()
        missing_datacenters = set(all_datacenters) - set(datacenter_stats['datacenter_id'])
        
        # Add missing datacenters with zero values
        if missing_datacenters:
            missing_df = pd.DataFrame({
                'datacenter_id': list(missing_datacenters),
                'used_slots': 0,
                'total_capacity': 0
            })
            datacenter_stats = pd.concat([datacenter_stats, missing_df], ignore_index=True)

        # Update used slots and total capacity in datacenter_capacity
        for _, row in datacenter_stats.iterrows():
            dc_max_slot_capacity = self.datacenter_capacity.loc[
                self.datacenter_capacity['datacenter_id'] == row['datacenter_id'], 
                'slots_capacity'
            ].iloc[0]  # Get the single value
            
            # Ensure capacity constraints are not violated
            if row['used_slots'] <= dc_max_slot_capacity:
                self.datacenter_capacity.loc[
                    self.datacenter_capacity['datacenter_id'] == row['datacenter_id'], 
                    ['used_slots', 'total_capacity']
                ] = [row['used_slots'], row['total_capacity']]
            else:
                # Convert solution to DataFrame and save as JSON
                solution_df = pd.DataFrame(self.solution)
                solution_df.to_json('./data/solution.json', orient='records', indent=4)
                raise ValueError(f"Datacenter '{row['datacenter_id']}': slot capacity exceeded")

    def update_time(self):
        self.time_step += 1

        # Update server lifespan and dismiss servers which are too old
        self.fleet = update_check_lifespan(self.fleet)


    def update_metrics(self, U=None, L=None, P=None):
        """
        Update performance metrics U, L, and P.

        Args:
            U (float, optional): Utilization metric.
            L (float, optional): Normalized lifespan metric.
            P (float, optional): Profit metric.

        Raises:
            ValueError: If no metrics are provided.
        """
        if U is None and L is None and P is None:
            raise ValueError("At least one metric (U, L, or P) must be provided.")
        
        if U is not None:
            self.performance_metrics['U'] = U
        if L is not None:
            self.performance_metrics['L'] = L
        if P is not None:
            self.performance_metrics['P'] = P


    def get_formatted_solution(self):
        return json.dumps(self.solution)
    
    @staticmethod
    def calculate_objective(fleet: pd.DataFrame, 
                            demand: pd.DataFrame, 
                            selling_prices: pd.DataFrame, 
                            time_step: int):
        """
        Calculate the combined objectives (U * L * P) for a given fleet, demand, and selling prices.

        Args:
            fleet (pd.DataFrame): Current server fleet information.
            demand (pd.DataFrame): Demand over time.
            selling_prices (pd.DataFrame): Selling price information.
            time_step (int): Current time step in the simulation.

        Returns:
            Product of utilization (U), normalized lifespan (L), and profit (P).
            Returns 0 if the fleet is empty.
        """
        if fleet.empty:
            return 0
        
        # Calculate U (utilization)
        D = get_time_step_demand(get_actual_demand(demand), time_step)
        Zf = get_capacity_by_server_generation_latency_sensitivity(fleet)
        U = get_utilization(D, Zf)

        # Calculate L (normalized lifespan)
        L = get_normalized_lifespan(fleet)

        # Calculate P (profit)
        selling_prices = change_selling_prices_format(selling_prices)
        P = get_profit(D, Zf, selling_prices, fleet)

        return U * L * P
