# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pyspark as spark

#read contents of csv into variables
datacenters = pd.read_csv("../data/datacenters.csv")
selling_prices = pd.read_csv("../data/selling_prices.csv")
servers = pd.read_csv("../data/servers.csv")

# %%
index_to_dcid = {0:"DC1",1:"DC2",2:"DC3",3:"DC4"}
generations = ['CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3']

#note demand_met is essentially min(zf, D)
columns = ['time_step', 'datacenter_id',
 'CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3', 
 'demand_met']

# %%
#checks if dc cap exceeded
def verify_solution_integrity(solution):
    dc_cap = datacenters["slots_capacity"].to_numpy()
    for i in range(4):
        datacenter = solution[solution["datacenter_id"] == index_to_dcid[i]].agg({j:"cumsum" for j in generations})
        dc = datacenter.to_numpy().astype("int")
        dc[95:] = dc[95:]-dc[:dc.shape[0]-95]
        cap_used = dc
        cap_used[:,0:4] = cap_used[:,0:4]*2
        cap_used[:,4:7] = cap_used[:,4:7]*4
        cap_used = np.sum(cap_used, axis=1)
        if(np.any(cap_used > dc_cap[i])):
            return False
    return True

# %%
 #make a long af integer value readable
def make_readable(value):
    string = ""
    value = str(int(value))
    for i in range(0,len(value)):
        if(i%3 ==0 and i != 0):
            string = ","+string
        string = value[len(value)-1-i]+string
    return string

def dict_gen(time_steps):
    d = {}
    for i in range(1,time_steps*4+1):
        d[i] = []
    return d

#convert the result array into a dataframe that contains the information
#the results are len(TIMESTEPS* numberofdatacenters* numberofservergens)
def result_to_df(result_x, result_y, start_step, time_steps):
    results_dict = dict_gen(time_steps)
    result_x = np.reshape(result_x,(time_steps, 4, 7))
    result_y = np.reshape(result_y,(time_steps, 4, 7))
    ts = start_step
    row_counter = 1
    #each row contains a timestep and certain certain dc's info
    for i in range(1,time_steps+1):
        for dc in range(4):
            results_dict[row_counter].append(ts)
            results_dict[row_counter].append(index_to_dcid[dc])
            total_bought = []
            for servergen in range(7):
                results_dict[row_counter].append(result_x[i-1, dc, servergen].solution_value())
                total_bought.append(result_y[i-1, dc, servergen].solution_value())
            results_dict[row_counter].append(total_bought)
            row_counter += 1
        ts+=1
    result_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=columns)
    return result_df

# %%
dc_cap = datacenters["slots_capacity"].to_numpy()
server_energies = servers["energy_consumption"].to_numpy()
purchase_prices = servers["purchase_price"].to_numpy()
capacity = servers["capacity"].to_numpy()

# demand3 = demand[demand["latency_sensitivity"] == "low"]
# demands = demand3.drop(columns=["latency_sensitivity","time_step"]).iloc[0:TIMESTEPS].to_numpy()

selling_prices_array = selling_prices[selling_prices["latency_sensitivity"] == "low"]["selling_price"].to_numpy()
maint_prices = servers["average_maintenance_fee"].to_numpy()
release_times = servers["release_time"].to_numpy()
index_to_dcid = {0:"DC1",1:"DC2",2:"DC3",3:"DC4"}

timestep_array = np.arange(1,97,1)
cpus1_maintenance_cost = servers[servers["server_generation"] == "CPU.S1"]["average_maintenance_fee"].iloc[0]
ts_array = 1.5 * timestep_array
maintenance_cost_array = np.empty((96,7))
for i in range(7):
    maintenance_cost_array[:,i] = (1+ ts_array/96 * np.log2(ts_array/96)) * maint_prices[i]

def profit(demand, x, y, TIMESTEPS, START_STEP):
    #x and y = shape(TIMESTEPS,DATACENTER,SERVERGEN)
    #get cumulative sum of number of servers for all servergens
    revenues = []
    costs = []
    #for each datacenter
    for datacenter in range(4):
        #get generated revenue at each timestep
        dc_id = index_to_dcid[datacenter]
        lat_sens = demand[demand["datacenter_id"] == dc_id]["latency_sensitivity"].iloc[0]
        dc_selling_prices = selling_prices[selling_prices["latency_sensitivity"] == lat_sens]["selling_price"].to_numpy()
        dc_revenues = []
        for i in range(TIMESTEPS):
            servergen = x[i, datacenter]
            #get demand met
            supply = y[i, datacenter]
            revenue = 0
            for j in range(7):
                revenue += supply[j] * dc_selling_prices[j].astype("int")
            dc_revenues.append(revenue)
            #print("revenues",revenue)
            # if(datacenter == 1):
            #     print(revenue)
        revenues.append(dc_revenues)
        #calc energycost for all servergens at the datacenter
        energy_costs = server_energies * datacenters[datacenters["datacenter_id"] == dc_id]["cost_of_energy"].to_numpy()

        timestep_costs = []
        for i in range(TIMESTEPS):
            #get servers that have been maintained (not new) for that datacenter
            #after a certain timeframe servers will have started to expire
            if(i>=96):
                maintained_servers = x[i-96:i, datacenter]
            else:
                maintained_servers = x[:i, datacenter]
            #calc cost of the new servers and add to overall cost at end
            new_cost = x[i, datacenter] * np.rint((purchase_prices + energy_costs + maintenance_cost_array[0])).astype("int")
            new_cost = np.sum(new_cost)
            #calc energy + maintenance cost
            energy_and_maint = maintenance_cost_array[:i] + energy_costs
            energy_and_maint = np.rint(energy_and_maint).astype("int")
            #multiply corresponding servers with their cost to get total for servergen at each ts
            maint_cost = np.sum(np.multiply(maintained_servers, energy_and_maint[:i][::-1]))
            # print("new:",new_cost)
            # print("maintained:",maint_cost)
            # if(maint_cost.size <= 0):
            #     maint_cost = np.zeros((7))
            if(maint_cost == 0):
                timestep_costs.append(new_cost)
            else:
                timestep_costs.append(maint_cost + new_cost)
            #print(maint_cost)
        costs.append(timestep_costs)
        

    #after all of the profits and costs have been calculated for all the datacenters at each timestep,
    #get sum of costs for the datacenters and the sum of profits for all datacenters at each timestep
    costs_sum = np.sum(costs, axis=0)
    revenue_sum = np.sum(revenues, axis=0)

    profit_arr = []
    #get profit at each timestep
    for i in range(TIMESTEPS):
        profit_arr.append(revenue_sum[i]-costs_sum[i])
    return profit_arr

def objective_func(demand, x, y, TIMESTEPS, START_STEP):
    x = np.reshape(x,(TIMESTEPS, 4, 7))
    y = np.reshape(y,(TIMESTEPS, 4, 7))
    P = profit(demand, x, y, TIMESTEPS, START_STEP)
    Objective = np.sum(P)
    return Objective

# %%
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp

def max_profit(demand, START_STEP, TIMESTEPS):
    demand2 = demand.merge(datacenters, on="latency_sensitivity", how="left")
    START_STEP2 = START_STEP
    TIMESTEPS2 = TIMESTEPS
    START_STEP %= 96
    TIMESTEPS %= 96
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("SAT")

    # Variables
    # x is the bought servergens at each timestep
    x = []
    # y is the min(supply, demand) at each timestep for each server
    y = []
    c = 0
    #makes an array of size (TIMESTEPS * dc_num * servergen_num)
    for i in range(TIMESTEPS2):
        #for all 4 datacenters
        for k in range(4):
            a = 0
            #generate cpu servers
            for j in range(4):
                x.append(solver.IntVar(0, int(dc_cap[k]/2), f'x{c}'))
                y.append(solver.IntVar(0, int((dc_cap[k]/2)*capacity[a]), f'y{c}'))
                a+=1
                c+=1
            #generate gpu servers
            for j in range(3):
                x.append(solver.IntVar(0, int(dc_cap[k]/4), f'x{c}'))
                y.append(solver.IntVar(0, int((dc_cap[k]/4)*capacity[a]), f'y{c}'))
                a+=1
                c+=1

    #print("Number of variables =", solver.NumVariables())

    # Constraints
    #adds constraint for retail time
    for k in range(4):
        start_pos = k*7*TIMESTEPS
        for i in range(7):
            rt = eval(release_times[i])
            counter = i
            for j in range(START_STEP2,START_STEP2+TIMESTEPS2):
                if(j < rt[0] or j > rt[1]):
                    solver.Add(x[start_pos+counter] == 0)
                counter+=7

    #get cumulative sum of the servergen at each timesteps
    cumsum_x = np.reshape(np.array(x), (TIMESTEPS2, 28))
    cumsum_x = np.cumsum(cumsum_x, axis=0)
    cumsum_x = np.reshape(cumsum_x, (TIMESTEPS2, 4, 7))

    for timestep in range(TIMESTEPS2):
        for datacenter in range(4):
            dc_id = index_to_dcid[datacenter]
            sens_demand = demand2[demand2["datacenter_id"] == dc_id].drop_duplicates(subset="time_step")
            #filter for the timesteps we need
            sens_demand = sens_demand[sens_demand["time_step"].isin(np.arange(START_STEP2, TIMESTEPS2+START_STEP2))]
            sens_demand = sens_demand.drop(columns=["time_step","datacenter_id","latency_sensitivity"]).to_numpy().astype("int")
            #total slots occupied cannot exceed dc capacity at any timeframe
            if(timestep>=96):
                cumsum_cpu_no_expired = np.subtract(cumsum_x[timestep][datacenter][:4], cumsum_x[timestep-96][datacenter][:4])
                cumsum_gpu_no_expired = np.subtract(cumsum_x[timestep][datacenter][4:], cumsum_x[timestep-96][datacenter][4:])
                solver.Add(np.sum(cumsum_cpu_no_expired*2)+np.sum(cumsum_gpu_no_expired*4)
                 <= dc_cap[datacenter])
            else:
                solver.Add(np.sum(cumsum_x[timestep][datacenter][:4]*2)+np.sum(cumsum_x[timestep][datacenter][4:]*4)
                 <= dc_cap[datacenter])
            for servergen in range(7):
                index = timestep*28+datacenter*7+servergen
                if(servergen < 4):
                    #dc capacity constraint for cpu
                    solver.Add(cumsum_x[timestep][datacenter][servergen]*2 <= dc_cap[datacenter])
                else:
                    #dc capacity constraint for gpu
                    solver.Add(cumsum_x[timestep][datacenter][servergen]*4 <= dc_cap[datacenter])
                if(datacenter == 2):
                    #dc3+dc4 demand should be less than sensdemand and less than their total cap
                    solver.Add(y[index]+y[timestep*28+3*7+servergen] <= sens_demand[timestep][servergen])
                    solver.Add(y[index]+y[timestep*28+3*7+servergen] <= 
                        cumsum_x[timestep][datacenter][servergen]*capacity[servergen]+cumsum_x[timestep][datacenter+1][servergen]*capacity[servergen])
                if(datacenter == 3):
                    solver.Add(y[index]+y[timestep*28+2*7+servergen] <= sens_demand[timestep][servergen])
                    solver.Add(y[index]+y[timestep*28+2*7+servergen] <= 
                        cumsum_x[timestep][datacenter][servergen]*capacity[servergen]+cumsum_x[timestep][datacenter-1][servergen]*capacity[servergen])
                else:
                    solver.Add(y[index] <= sens_demand[timestep][servergen])
                    solver.Add(y[index] <= cumsum_x[timestep][datacenter][servergen]*capacity[servergen])

    #print("Number of constraints =", solver.NumConstraints())

    # Objective
    solver.Maximize(objective_func(demand2, x, y, TIMESTEPS2, START_STEP))
    
    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Total value =', make_readable(solver.Objective().Value()))
        # for i in range(7):
        #     print(f'Item {i}: {x[i].solution_value()}')
    else:
        print('The problem does not have an optimal solution.')

    if status == pywraplp.Solver.OPTIMAL:
        result_df = result_to_df(x,y,START_STEP2,TIMESTEPS2)
    else:
        return "no solution"
    
    return result_df

# %%
# result_df = max_profit(pd.read_csv("../data/demand.csv"), 1, 167)
# print(result_df)
# valid = verify_solution_integrity(result_df)
# if(not valid):
#     print("solution has an error!")
