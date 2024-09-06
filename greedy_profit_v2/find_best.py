import os
import re
import json

def find_best_scores(base_dir):
    best_scores = {}

    # Traverse the directory structure
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'score.txt':
                config = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    content = f.read()
                    matches = re.findall(r'Solution score for (\d+): ([\d.]+)', content)
                    
                    for match in matches:
                        seed = int(match[0])
                        score = float(match[1])
                        
                        if seed not in best_scores or score > best_scores[seed]['best_score']:
                            best_scores[seed] = {
                                'best_score': score,
                                'best_configuration': config
                            }
    
    return best_scores

base_dir = 'greedy_profit_v2/output_test'
best_scores = find_best_scores(base_dir)

# Calculate the mean score
total_score = sum(entry['best_score'] for entry in best_scores.values())
best_scores['mean_score'] = total_score / len(best_scores)

# save to json
with open('best_scores.json', 'w') as f:
    json.dump(best_scores, f, indent=4)