from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds
import numpy as np

 #make a long af integer value readable
def make_readable(value):
    string = ""
    value = str(int(value))
    for i in range(0,len(value)):
        if(i%3 ==0 and i != 0):
            string = ","+string
        string = value[len(value)-1-i]+string
    return string

seeds = known_seeds()

scores = []

for seed in seeds:

    # LOAD SOLUTION
    fleet, pricing_strategy = load_solution('output/'+str(seed)+'.json')

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(fleet,
                                pricing_strategy,
                                demand,
                                datacenters,
                                servers,   
                                selling_prices,
                                elasticity,
                                seed=seed,
                                verbose=0)
    print(make_readable(score))
    scores.append(score)

scores = np.array(scores)
mean_score = np.mean(scores)
mean_score = make_readable(mean_score)
print(f'Solution score: {mean_score}')