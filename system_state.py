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
        objective (dict): Dictionary storing current U, L, and P metrics.
        solution (list): List of all actions taken in the simulation.
    """


    def __init__(self, datacenters: pd.DataFrame, servers: pd.DataFrame):
        self.time_step = 1
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
            'cost_of_moving':          pd.Series(dtype='int'),
            'moved':                   pd.Series(dtype='int')
        })
        
        # Track datacenters' slot capacity
        self.datacenter_capacity = datacenters[['datacenter_id', 'slots_capacity']].copy()
        self.datacenter_capacity['used_slots'] = 0
        self.datacenter_capacity['total_capacity'] = 0

        self.objective = {'time-step': self.time_step, 'O': 0, 'U': 0, 'L': 0, 'P': 0}
        self.solution = []

        # Convert the 'release_time' column from string to list
        servers['release_time'] = pd.eval(servers['release_time'])

        self.servers_info = servers
        self.datacenter_info = datacenters


    def __str__(self) -> str:
        obj = self.objective.copy()
        obj['O'] = round(self.objective['O'], 2)
        return str(obj)


    def update_state(self, decision):
        """
        OPTIONAL

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
        dismiss_server_ids = set()
        move_decisions = []

        for decision in decisions:
            if decision['action'] == 'buy':
                buy_decisions.append(decision)
            elif decision['action'] == 'dismiss':
                dismiss_server_ids.add(decision['server_id'])
            elif decision['action'] == 'move':
                move_decisions.append(decision)

        if buy_decisions:
            buy_df = pd.DataFrame(buy_decisions)
            
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
            
            # Create DataFrame for new servers
            new_servers = pd.DataFrame({
                'datacenter_id':           buy_df['datacenter_id'],
                'server_generation':       buy_df['server_generation'],
                'server_id':               buy_df['server_id'],
                'lifespan':                1,
                'life_expectancy':         buy_df['life_expectancy'],
                'latency_sensitivity':     buy_df['latency_sensitivity'],
                'capacity':                buy_df['capacity'],
                'purchase_price':          buy_df['purchase_price'],
                'average_maintenance_fee': buy_df['average_maintenance_fee'],
                'energy_consumption':      buy_df['energy_consumption'],
                'cost_of_energy':          buy_df['cost_of_energy'],
                'cost_of_moving':          buy_df['cost_of_moving'],
                'moved':                   0
            })

            self.fleet = pd.concat([self.fleet, new_servers], ignore_index=True)


        if dismiss_server_ids:
            self.fleet = self.fleet[~self.fleet['server_id'].isin(dismiss_server_ids)]

        if move_decisions:
            # Create mapping of server_id to new datacenter_id
            new_dc_map = { d['server_id']: d['datacenter_id'] for d in move_decisions }
            latency_map = dict(zip(self.datacenter_info['datacenter_id'], self.datacenter_info['latency_sensitivity']))

            # Servers to be moved
            move_mask = self.fleet['server_id'].isin(new_dc_map.keys())

            # Update datacenter_id for moved servers
            dc_ids = self.fleet.loc[move_mask, 'server_id'].map(new_dc_map)
            latency_sensitivities = dc_ids.map(latency_map)

            self.fleet.loc[move_mask, 'datacenter_id'] = dc_ids
            self.fleet.loc[move_mask, 'latency_sensitivity'] = latency_sensitivities
            self.fleet.loc[move_mask, 'moved'] = 1


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
                solution_df.to_json('error_actions.json', orient='records', indent=4)
                print(datacenter_stats)
                raise ValueError(f"Datacenter '{row['datacenter_id']}': slot capacity exceeded")

    def update_time(self, ts=1):
        self.time_step += ts

        # Pandas vectorized operation
        # thx Jamie for reminder
        self.fleet['lifespan'] += ts

        # OPTIONAL
        # Update server lifespan and dismiss servers which are too old
        # self.fleet = update_check_lifespan(self.fleet)


    def update_objective(self, demand, selling_prices):
        """
        Calculate and update the objectives O, U, L and P for a given fleet, 
            demand at time step, and selling prices.

        Args:
            demand (pd.DataFrame): Demand over time.
            selling_prices (pd.DataFrame): Selling price information.

        Returns:
            Product of utilization (U), normalized lifespan (L), and profit (P).
            Returns 0 if the fleet is empty.
        """
        
        # Calculate U (utilizatio n)
        D = get_time_step_demand(demand, self.time_step)
        Zf = get_capacity_by_server_generation_latency_sensitivity(self.fleet)
        U = get_utilization(D, Zf)

        # Calculate L (normalized lifespan)
        L = get_normalized_lifespan(self.fleet)

        # Calculate P (profit)
        selling_prices = change_selling_prices_format(selling_prices)
        P = get_profit(D, Zf, selling_prices, self.fleet)
        
        # Calculate O (objective)
        obj = U * L * P
        O = self.objective['O'] + obj
        
        # Update
        self.objective['time-step'] = self.time_step
        self.objective['O'] = O
        self.objective['U'] = round(U, 2)
        self.objective['L'] = round(L, 2)
        self.objective['P'] = round(P, 2)
