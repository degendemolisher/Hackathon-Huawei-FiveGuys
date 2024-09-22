from seeds import known_seeds
import json
import pandas as pd

seeds = known_seeds()
# read from solution json
with open(f"./output/{seeds[0]}.json", "r") as f:
    solution = json.load(f)
    fleet = solution["fleet"]
    pricing_strategy = solution["pricing_strategy"]

    # export pricing strat to df
    pricing_strategy = pd.DataFrame(pricing_strategy)
    print(pricing_strategy)