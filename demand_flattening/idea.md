### Summary

The ability to change price implies we also have the ability to change the demand, so we can adjust the demand to negate the negative effects of having large troughs/peaks of unmet/excess demand.

### Before the algorithm

PoC file: pre_flattening.py

1) Produce moving average of the demand with window size w
2) Run some algorithm which buys servers for a fixed set of moving average demand produced at step 1 for all time steps
3) For each latency sensitivity/server generation
    1) Calculate the difference between the moving average demand and the actual demand for all time steps (delta D)
    2) Given elasticity, calculate the price change (delta p) required to bring actual demand to the moving average for all time steps
    3) Calculate the new price for all time steps
    4) Convert the new prices into price change actions with time step, latency sensitivity, and server generation
    
### After the algorithm

PoC file: post_flattening.py

1) Run some algorithm which buys servers for a fixed set of actual demand for all time steps
2) Calculate the capacity at all time steps for each server generation and latency sensitivty
3) For each latency sensitivity/server generation
    1) Calculate the difference between the demand met and the actual demand for all time steps (delta D)
    2) Given elasticity, calculate the delta p for all time steps
    3) Calculate the new price for all time steps
    4) Convert the new prices into price change actions with time step, latency sensitivity and server generation