from utils import (load_problem_data,
                   load_solution)
from evaluation_v6 import evaluation_function


# LOAD SOLUTION
solution = load_solution('gpuall_merge_threshold_x4.json')

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(solution,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            seed=123,
                            verbose=True)

print(f'Solution score: {score}')