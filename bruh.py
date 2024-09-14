import json

# Load the JSON data
with open('best_scores.json', 'r') as f:
    data = json.load(f)

# Transform the data
transformed_data = {}
for seed, details in data.items():
    if seed == 'mean_score':
        transformed_data['mean_score'] = details
    else:
        config = details['best_configuration']
        quantile, range_multiplier = config.split('_merge_')
        transformed_data[seed] = {
            'quantile': quantile,
            'range_multiplier': range_multiplier
        }

# Save the transformed JSON
with open('best_scores_transformed.json', 'w') as f:
    json.dump(transformed_data, f, indent=4)