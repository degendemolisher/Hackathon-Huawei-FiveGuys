# TODO: Optimisation idea
# For each reused `function()`, replace it with `cache.function()`, and the initial result will be stored in this file

import pandas as pd
import evaluation


_current_fleet_capacity = False

def get_capacity_by_server_generation_latency_sensitivity(fleet: pd.DataFrame) -> pd.DataFrame:
    if _current_fleet_capacity == False:
        _current_fleet_capacity = evaluation.get_capacity_by_server_generation_latency_sensitivity(fleet)
    return _current_fleet_capacity

def reset():
    _current_fleet_capacity = False