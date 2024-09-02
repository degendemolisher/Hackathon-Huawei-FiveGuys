import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pyspark as spark

#read contents of csv into variables
datacenters = pd.read_csv("../data/datacenters.csv")
demand = pd.read_csv("../data/demand.csv")
selling_prices = pd.read_csv("../data/selling_prices.csv")
servers = pd.read_csv("../data/servers.csv")

index_to_dcid = {0:"DC1",1:"DC2",2:"DC3",3:"DC4"}

columns = ['time_step', 'datacenter_id',
 'CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3', 
 'total_owned']

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
def result_to_df(result_x, result_y, time_steps):
    results_dict = dict_gen(time_steps)
    result_x = np.reshape(result_x,(time_steps, 4, 7))
    result_y = np.reshape(result_y,(time_steps, 4, 7))
    row_counter = 1
    #each row contains a timestep and certain certain dc's info
    for i in range(1,time_steps+1):
        for dc in range(4):
            results_dict[row_counter].append(i)
            results_dict[row_counter].append(index_to_dcid[dc])
            total_bought = []
            for servergen in range(7):
                results_dict[row_counter].append(result_x[i-1, dc, servergen].solution_value())
                total_bought.append(result_y[i-1, dc, servergen].solution_value())
            results_dict[row_counter].append(total_bought)
            row_counter += 1
    result_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=columns)
    return result_df

import numpy as np
from scipy.optimize import minimize

#for DC1 and servergen cpus1 over first TIMESTEPS timesteps

TIMESTEPS = 24
START_STEP = 1

dc_cap = datacenters["slots_capacity"].to_numpy()
server_energies = servers["energy_consumption"].to_numpy()
purchase_prices = servers["purchase_price"].to_numpy()
capacity = servers["capacity"].to_numpy()

cpus1_energy= servers[servers["server_generation"] == "CPU.S1"]["energy_consumption"].iloc[0]
cpus1_purchasecost = servers[servers["server_generation"] == "CPU.S1"]["purchase_price"].iloc[0]

demand2 = demand.merge(datacenters, on="latency_sensitivity", how="left")
demand3 = demand[demand["latency_sensitivity"] == "low"]
demands = demand3.drop(columns=["latency_sensitivity","time_step"]).iloc[0:TIMESTEPS].to_numpy()

selling_prices_array = selling_prices[selling_prices["latency_sensitivity"] == "low"]["selling_price"].to_numpy()
maint_prices = servers["average_maintenance_fee"].to_numpy()
release_times = servers["release_time"].to_numpy()
index_to_dcid = {0:"DC1",1:"DC2",2:"DC3",3:"DC4"}

timestep_array = np.arange(1,96,1)
cpus1_maintenance_cost = servers[servers["server_generation"] == "CPU.S1"]["average_maintenance_fee"].iloc[0]
ts_array = 1.5 * timestep_array
maintenance_cost_array = np.empty((95,7))
for i in range(7):
    maintenance_cost_array[:,i] = (1+ ts_array/96 * np.log2(ts_array/96)) * maint_prices[i]
epsilon = 0.00000001

#where x is an array containing what servergen was bought at each timestep for all servergens
def capacity_constraint(x):
    x = np.reshape(x,(TIMESTEPS,7))
    total = 0
    #servernumber * slotsize to get slots occupied
    #for cpu
    occupied_slots = np.sum(x[:,0:4] * 2)
    #for gpu
    total = occupied_slots + np.sum(x[:,4:7] * 4)
    # #get total number of servers purchased
    # total = np.sum(x)
    # #servernumber * slotsize to get slots occupied
    # total = total * 2
    #constraint used cap has to be less than dc1_cap
    return dc1_cap - total

#get utilisation over the timesteps
def utilisation(x, y):
    #print(x)
    #array of utilisation of each server at eachtimestep
    util = []
    for i in range(TIMESTEPS):
        #array of bought servergens at this timestep
        servergen = x[i]
        #array of min(servergen_supply,demand) at this timestep for all servergens
        s_d_min = y[i]
        #get cumulative sum of number of servers to get total owned at each timestep
        cumsum = []
        total = epsilon
        #calc number of servers at each timestep and their cap
        s_d_sum = []
        sum = 0
        for j in range(len(servergen)):
            sum += s_d_min[j]
            cumsum.append((total + servergen[j]) * capacity[j])
            total = cumsum[j]
        #cumsum = np.cumsum(servergen)
        #get their capacity
        #get demand met for servergen i
        #get fraction of servergen utilised at each timestep
        util.append(sum)

    return util, cumsum

def lifespan(x,y):
    #get number of servers bought for timesteps (servergen doesnt matter)
    ts_sum = np.sum(x, axis=1)
    cumsum = np.cumsum(ts_sum)+epsilon
    life_spans = []
    for i in range(1,TIMESTEPS+1):
        multiplication_arr = np.arange(i,0,-1)
        #array[i-1] = np.divide(np.sum(np.multiply(ts_sum[0:i], multiplication_arr[0:i])), 96)
        sum = 0
        for j in range(i):
            m = ts_sum[j] * multiplication_arr[j]
            sum += m
        life_spans.append(sum)

    return life_spans, cumsum

def profit(x, y):
    #x and y = shape(TIMESTEPS,DATACENTER,SERVERGEN)
    #get cumulative sum of number of servers for all servergens
    revenues = []
    costs = []
    #for each datacenter
    for datacenter in range(4):
        #get generated revenue at each timestep
        dc_id = index_to_dcid[datacenter]
        lat_sens = demand2[demand2["datacenter_id"] == dc_id]["latency_sensitivity"].iloc[0]
        dc_selling_prices = selling_prices[selling_prices["latency_sensitivity"] == lat_sens]["selling_price"].to_numpy()
        dc_revenues = []
        for i in range(TIMESTEPS):
            servergen = x[i, datacenter, :]
            #get demand met
            supply = y[i, datacenter, :]
            revenue = 0
            for j in range(7):
                revenue += supply[j] * dc_selling_prices[j].astype("int") * capacity[datacenter]
            dc_revenues.append(revenue)
        revenues.append(dc_revenues)
        #calc energycost for all servergens at the datacenter
        energy_costs = server_energies * datacenters[datacenters["datacenter_id"] == dc_id]["cost_of_energy"].to_numpy()

        timestep_costs = []
        for i in range(TIMESTEPS):
            #get servers that have been maintained (not new) for that datacenter
            maintained_servers = x[:i, datacenter]
            #calc cost of the new servers and add to overall cost at end
            new_cost = x[i, datacenter] * np.rint((purchase_prices + energy_costs + maintenance_cost_array[i])).astype("int")
            new_cost = np.sum(new_cost)
            #calc energy + maintenance cost
            energy_and_maint = maintenance_cost_array[:i] + energy_costs
            energy_and_maint = np.rint(energy_and_maint).astype("int")
            #multiply corresponding servers with their cost to get total for servergen at each ts
            maint_cost = np.sum(np.multiply(maintained_servers, energy_and_maint[:i]))
            # if(maint_cost.size <= 0):
            #     maint_cost = np.zeros((7))
            timestep_costs.append(maint_cost + new_cost)
        costs.append(timestep_costs)

    #after all of the profits and costs have been calculated for all the datacenters at each timestep,
    #get sum of costs for the datacenters and the sum of profits for all datacenters at each timestep
    costs_sum = np.sum(costs, axis=0)
    revenue_sum = np.sum(revenues, axis=0)
    profit_arr = []
    #get profit at each timestep
    for i in range(TIMESTEPS):
        profit_arr.append(revenue_sum[i]-costs_sum[i])
    # print(revenues[0])
    # print(len(timestep_costs))
    return profit_arr

    # ts1x = x[0]
    # ts2x = x[1]
    # ts1revenue = min(60*ts1x, 4000) * 10
    # ts2revenue = min(60*ts1x+60*ts2x, 8160) * 10

    # maint_cost = maintenance_cost(x)
    # ts1cost = (1500 + energycost + maint_cost[0]) * ts1x
    # ts2cost = (energycost + maint_cost[1]) * ts1x + (energycost + maint_cost[0]) * ts2x

    # ts1p = ts1revenue-ts1cost
    # ts2p = ts2revenue-ts2cost
    # return [ts1p, ts2p]

def maintenance_cost(x):
    return maintenance_cost_array[0:len(x)]
    # ts1x = x[0]
    # ts2x = x[1]

    # ts1maintenance = (1+ 1.5/96 * np.log2(1.5/96)) * cpus1_maintenance_cost
    # ts2maintenance = (1+ 3/96 * np.log2(3/96)) * cpus1_maintenance_cost
    # return [ts1maintenance, ts2maintenance]


def objective_func(x, y):
    x = np.reshape(x,(TIMESTEPS, 4, 7))
    y = np.reshape(y,(TIMESTEPS, 4, 7))
    P = profit(x, y)
    Objective = np.sum(P)
    return Objective

from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp

def max_profit():
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("SAT")

    # Variables
    # x is the bought servergens at each timestep
    x = []
    # y is the min(supply, demand) at each timestep for each server
    y = []
    c = 0
    #makes an array of size (TIMESTEPS * dc_num * servergen_num)
    for i in range(TIMESTEPS):
        #for all 4 datacenters
        for k in range(4):
            #generate cpu servers
            for j in range(4):
                x.append(solver.IntVar(0, int(dc_cap[k]/2), f'x{c}'))
                y.append(solver.IntVar(0, int(dc_cap[k]/2), f'y{c}'))
                c+=1
            #generate gpu servers
            for j in range(3):
                x.append(solver.IntVar(0, int(dc_cap[k]/4), f'x{c}'))
                y.append(solver.IntVar(0, int(dc_cap[k]/4), f'y{c}'))
                c+=1
    # z is the accumulated number of servers at each timestep
    #z = [solver.IntVar(0, int(dc1_cap/2), f'z{i}') for i in range(TIMESTEPS*7)]
    print("Number of variables =", solver.NumVariables())

    # Constraints
    #adds constraint for retail time
    for k in range(4):
        start_pos = k*7*TIMESTEPS
        for i in range(7):
            rt = eval(release_times[i])
            counter = i
            for j in range(START_STEP,START_STEP+TIMESTEPS):
                if(j < rt[0] or j > rt[1]):
                    solver.Add(x[start_pos+counter] == 0)
                counter+=7

    total_x = [0,0,0,0,0,0,0]
    for timestep in range(TIMESTEPS):
        for datacenter in range(4):
            dc_id = index_to_dcid[datacenter]
            sens_demand = demand2[demand2["datacenter_id"] == dc_id].drop_duplicates(subset="time_step").to_numpy()
            for servergen in range(7):
                index = timestep*28+datacenter*7+servergen
                total_x[servergen] += x[index]
                if(servergen < 4):
                    #dc capacity constraint
                    solver.Add(total_x[servergen]*2 <= dc_cap[datacenter])
                else:
                    #dc capacity constraint
                    solver.Add(total_x[servergen]*4 <= dc_cap[datacenter])
                #FIX DEMANDS ASWELLLLL!!!!!!
                solver.Add(y[index] <= demands[timestep][servergen])
                solver.Add(y[index] <= total_x[servergen]*60)

    print("Number of constraints =", solver.NumConstraints())

    # Objective
    solver.Maximize(objective_func(x, y))

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Total value =', make_readable(solver.Objective().Value()))
        for i in range(7):
            print(f'Item {i}: {x[i].solution_value()}')
    else:
        print('The problem does not have an optimal solution.')

    if status == pywraplp.Solver.OPTIMAL:
        result_df = result_to_df(x,y,TIMESTEPS)
    else:
        return "no solution"
    
    return result_df

result_df = max_profit()
print(result_df)