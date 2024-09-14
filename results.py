import json
import uuid


def save_results_as_actions(path: str, results: list[dict]):
    """
    Converts the results from a list of dictionaries of format:
    [{
        'server_generation': str,
        'buy_count': int,
        'datacenter_id': str,
        'buy_time_step': int,
        'dismiss_time_step': int
    }]
    to:
    [{
        'action': 'buy',
        'server_generation': str,
        'datacenter_id': str,
        'time_step': int,
        'server_id': str
    },
    {
        'action': 'dismiss',
        'server_generation': str,
        'datacenter_id': str,
        'time_step': int,
        'server_id': str
    }]
    """
    
    actions = []
    for entry in results:
        server_gen = entry['server_generation']
        buy_count = int(entry['buy_count'])
        datacenter_id = entry['datacenter_id']
        buy_time_step = int(entry['buy_time_step'])
        dismiss_time_step = int(entry['dismiss_time_step'])
        
        # the loop doesn't work for some reason, use commented code below
        for _ in range(0, buy_count):
            server_id = str(uuid.uuid4())
            buy_entry = {
                'action': 'buy',
                'server_generation': server_gen,
                'datacenter_id': datacenter_id,
                'time_step': buy_time_step,
                'server_id': server_id
            }
            actions.append(buy_entry)

            # Check if the server will automatically expire before it is dismissed
            if (dismiss_time_step - buy_time_step) > 96:
                continue

            # Dismiss time step may be 169 if demand lasts to final time step
            # So ignore all dismiss time steps greater than 168
            if (dismiss_time_step <= 168):
                dismiss_entry = {
                    'action': 'dismiss',
                    'server_generation': server_gen,
                    'datacenter_id': datacenter_id,
                    'time_step': dismiss_time_step,
                    'server_id': server_id
                }
                actions.append(dismiss_entry)

    # sort by time step
    actions.sort(key=lambda x: x['time_step'])

    # write to json
    with open(path, 'w') as f:
        json.dump(actions, f, indent=4)