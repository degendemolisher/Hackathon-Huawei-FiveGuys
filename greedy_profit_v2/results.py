import json
import uuid

def save_results_as_actions(path: str, results: dict):
    """
    Converts the results from a dictionary of format:
    {
        "fleet": [
            {
                'server_generation': str,
                'buy_count': int,
                'datacenter_id': str,
                'buy_time_step': int,
                'dismiss_time_step': int
            }
        ],
        "pricing_strategy": [
            {
                'time_step': int,
                'latency_sensitivity': str,
                'server_generation': str,
                'price': float
            }
        ]
    }
    to:
    {
        "fleet": [
            {
                'time_step': int,
                'datacenter_id': str,
                'server_generation': str,
                'server_id': str,
                'action': str
            }
        ],
        "pricing_strategy": [
            {
                'time_step': int,
                'latency_sensitivity': str,
                'server_generation': str,
                'price': float
            }
        ]
    }
    """
    
    fleet_actions = []
    
    # Process fleet data
    for entry in results['fleet']:
        server_gen = entry['server_generation']
        buy_count = int(entry['buy_count'])
        datacenter_id = entry['datacenter_id']
        buy_time_step = int(entry['buy_time_step'])
        dismiss_time_step = int(entry['dismiss_time_step'])
        
        for _ in range(0, buy_count):
            server_id = str(uuid.uuid4())
            buy_entry = {
                'time_step': buy_time_step,
                'datacenter_id': datacenter_id,
                'server_generation': server_gen,
                'server_id': server_id,
                'action': 'buy'
            }
            fleet_actions.append(buy_entry)

            if (dismiss_time_step - buy_time_step) > 96:
                continue

            if (dismiss_time_step <= 168):
                dismiss_entry = {
                    'time_step': dismiss_time_step,
                    'datacenter_id': datacenter_id,
                    'server_generation': server_gen,
                    'server_id': server_id,
                    'action': 'dismiss'
                }
                fleet_actions.append(dismiss_entry)


    if len(results['pricing_strategy']) == 0:
        results['pricing_strategy'] = [
            {
                'time_step': 1,
                'latency_sensitivity': 'low',
                'server_generation': 'CPU.S1',
                'price': 10,
            }
        ]
    # Combine fleet and pricing strategy actions
    combined_results = {
        "fleet": fleet_actions,
        "pricing_strategy": results['pricing_strategy']
    }

    # Write to json
    with open(path, 'w') as f:
        json.dump(combined_results, f, indent=4)