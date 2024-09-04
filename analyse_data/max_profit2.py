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
def verify_solution_integrity(solution, should_print=False):
    dc_cap = datacenters["slots_capacity"].to_numpy()
    for i in range(4):
        datacenter = solution[solution["datacenter_id"] == index_to_dcid[i]].agg({j:"cumsum" for j in generations})
        dc = datacenter.to_numpy().astype("int")
        # if(solution["time_step"].max()-solution["time_step"].min() >= 96):
        #     dc[95:] = dc[95:]-dc[dc.shape[0]-95]
        for j in range(dc[95:].shape[0]):
            dc[95+j] = dc[95+j]-dc[j]
        #print(dc)
        cap_used = dc
        cap_used[:,0:4] = cap_used[:,0:4]*2
        cap_used[:,4:7] = cap_used[:,4:7]*4
        cap_used = np.sum(cap_used, axis=1)
        # if(should_print):
        #     print(cap_used)
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
def result_to_df(result_x, result_y, step_size, start_step, time_steps):
    results_dict = dict_gen(time_steps)
    chunks = int(time_steps/step_size)
    result_x = np.reshape(result_x,(chunks, 4, 7))
    result_y = np.reshape(result_y,(time_steps, 3, 7))
    ts = start_step
    row_counter = 1
    count = 0
    #each row contains a timestep and certain certain dc's info
    for i in range(1,time_steps+1):
        chunk = int((i-1)/step_size)
        for dc in range(4):
            #print(chunk)
            results_dict[row_counter].append(ts)
            results_dict[row_counter].append(index_to_dcid[dc])
            total_supply = []
            for servergen in range(7):
                if((i-1)%step_size==0):
                    results_dict[row_counter].append(int(result_x[chunk, dc, servergen].solution_value()))
                    count+=1
                else:
                    results_dict[row_counter].append(0)
                if(dc == 3):
                    total_supply.append(int(result_y[i-1, dc-1, servergen].solution_value()))
                else:
                    total_supply.append(int(result_y[i-1, dc, servergen].solution_value()))
            results_dict[row_counter].append(total_supply)
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

def profit(demand, x, y, step_size, TIMESTEPS, START_STEP):
    chunk = TIMESTEPS/step_size
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
        #skip dc4 as y for dc3 already has the info for its demand met
        if(datacenter != 3):
            for i in range(TIMESTEPS):
                #get demand met
                supply = y[i, datacenter]
                revenue = 0
                for j in range(7):
                    revenue += supply[j] * dc_selling_prices[j].astype("int")
                dc_revenues.append(revenue)
            revenues.append(dc_revenues)
        #calc energycost for all servergens at the datacenter
        energy_costs = server_energies * datacenters[datacenters["datacenter_id"] == dc_id]["cost_of_energy"].to_numpy()

        timestep_costs = []
        mult_array = np.empty((95,7))
        for i in range(7):
            mult_array[:,i] = np.arange(2,97)/96
        #starts at 2 as maintained servers start from 2life
        #REMINDER:can do buy action every step_size steps
        #get cost equation at each timestep
        chunks = int(TIMESTEPS/step_size)
        for i in range(chunks):
            for step in range(step_size):
                #xshape=(chunks,dc,servergen)
                #get servers that have been maintained (not new) for that datacenter
                #calc cost of the new servers and add to overall cost at end
                #if buy action was performable on this timestep:
                #curr_chunk = int(i/step_size)
                timestep = i*step_size+step
                if(step==0):
                    #after a certain timeframe servers will have started to expire
                    if(timestep>=96):
                        maintained_servers = x[i%7:i, datacenter]
                    else:
                        maintained_servers = x[:i, datacenter]
                    new_cost = x[i, datacenter] * np.rint((purchase_prices + energy_costs + maintenance_cost_array[0])*1/96).astype("int")
                    new_cost = np.sum(new_cost)
                    #calc energy + maintenance cost
                    energy_and_maint = maintenance_cost_array[1:timestep+1] + energy_costs
                    energy_and_maint = np.rint(energy_and_maint).astype("int")
                    
                    #multiply corresponding servers with their cost to get total for servergen at each ts
                    if(timestep>=96):
                        maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep-1:step_size][:7][::-1])
                        #maint_cost = np.multiply(maint_cost, mult_array[step:timestep:step_size][:7][::-1])
                        maint_cost = np.sum(maint_cost)
                    else:
                        maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep-1:step_size][::-1])
                        #maint_cost = np.multiply(maint_cost, mult_array[step:timestep:step_size][::-1])
                        maint_cost = np.sum(maint_cost)

                    if(maint_cost == 0):
                        timestep_costs.append(new_cost)
                    else:
                        timestep_costs.append(maint_cost + new_cost)
                else:
                    if(timestep>=96):
                        maintained_servers = x[i%7:i+1, datacenter]
                    else:
                        maintained_servers = x[:i+1, datacenter]
                    #calc energy + maintenance cost
                    energy_and_maint = maintenance_cost_array[:timestep+1] + energy_costs
                    energy_and_maint = np.rint(energy_and_maint).astype("int")
                    #multiply corresponding servers with their cost to get total for servergen at each ts
                    maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep+step_size:step_size][::-1])
                    maint_cost = np.multiply(maint_cost, mult_array[step-1:timestep+step_size-1:step_size][::-1])
                    #(maint_cost)
                    maint_cost = np.sum(maint_cost)
                    # if(timestep<14 and datacenter==0):
                    #     print("m:",maint_cost)
                    timestep_costs.append(maint_cost)
            # if(i>6 and i<9 and datacenter == 0):
            #     print(timestep)
            #     print(maint_cost)
        costs.append(timestep_costs)
        

    #after all of the profits and costs have been calculated for all the datacenters at each timestep,
    #get sum of costs for the datacenters and the sum of profits for all datacenters at each timestep
    # print("c1: ",costs[0][0])
    # print("c2: ",costs[0][1])
    # print("c3: ",costs[0][3])
    # print("c4: ",costs[0][4])
    costs_sum = np.sum(costs, axis=0)
    # print("c1",costs_sum[0])
    # print()
    # print("c2",costs_sum[10])
    # print(costs_sum.shape)
    revenue_sum = np.sum(revenues, axis=0)

    profit_arr = []
    #get profit at each timestep
    for i in range(TIMESTEPS):
        profit_arr.append(revenue_sum[i]-costs_sum[i])
    return profit_arr

def objective_func(demand, x, y, step_size, TIMESTEPS, START_STEP):
    chunk = int(TIMESTEPS/step_size)
    x = np.reshape(x,(chunk, 4, 7))
    y = np.reshape(y,(TIMESTEPS, 3, 7))
    P = profit(demand, x, y, step_size, TIMESTEPS, START_STEP)
    Objective = np.sum(P)
    return Objective

# %%
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp

def max_profit(demand, step_size=12, START_STEP=1, TIMESTEPS=168):
    demand2 = demand.merge(datacenters, on="latency_sensitivity", how="left")
    START_STEP2 = START_STEP
    TIMESTEPS2 = TIMESTEPS
    START_STEP %= 96
    TIMESTEPS %= 96
    chunks = int(TIMESTEPS2/step_size)
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("CLP")

    # Variables
    # x is the bought servergens at each timestep chunk/checkpoint
    x = []
    # y is the min(supply, demand) at each timestep for each server SUPPLYING TO EACH LATENCY its shape is 
    #(chunks, latencies, servergens)
    y = []
    # r will be the same as y, but with the constraints modified to account for serverlifespan
    r = []
    #BASIC Z IMPLEMENTATION FOR NOW
    # z is the number of servers to dismiss, shape is same
    z = []
    c = 0
    #makes an array of size (chunks * dc_num * servergen_num)
    for i in range(chunks):
        #for all 4 datacenters
        for k in range(4):
            a = 0
            #generate cpu servers
            for j in range(4):
                x.append(solver.IntVar(0, int(dc_cap[k]/2), f'x{c}'))
                #z.append(solver.IntVar(0, int(dc_cap[k]/2), f'z{c}'))
                a+=1
                c+=1
            #generate gpu servers
            for j in range(3):
                x.append(solver.IntVar(0, int(dc_cap[k]/4), f'x{c}'))
                #z.append(solver.IntVar(0, int(dc_cap[k]/4), f'z{c}'))
                a+=1
                c+=1
    c=0
    #makes an array of size (timesteps * dc_num * servergen_num)
    for i in range(TIMESTEPS2):
        #for all 4 datacenters
        for k in range(4):
            a = 0
            #generate cpu servers
            for j in range(4):
                if(k != 3):
                    r.append(solver.IntVar(0, int((dc_cap[k]/2)*capacity[a]), f'r{c}'))
                a+=1
                c+=1
            #generate gpu servers
            for j in range(3):
                if(k !=3):
                    r.append(solver.IntVar(0, int((dc_cap[k]/4)*capacity[a]), f'r{c}'))
                a+=1
                c+=1

    #print("Number of variables =", solver.NumVariables())

    # Constraints
    #adds constraint for retail time
    for k in range(4):
        action_time_array = np.arange(1,TIMESTEPS,step_size)
        start_pos = k*7*chunks
        for i in range(7):
            rt = eval(release_times[i])
            counter = i
            for j in action_time_array:
                if(j < rt[0] or j > rt[1]):
                    solver.Add(x[start_pos+counter] == 0)
                counter+=7

    #get cumulative sum of the servergen at each timesteps
    cumsum_x = np.reshape(np.array(x), (chunks, 28))
    cumsum_x = np.cumsum(cumsum_x, axis=0)
    cumsum_x = np.reshape(cumsum_x, (chunks, 4, 7))

    mult_array = np.empty(((96),7))
    counter=0
    for i in range(1,96+1):
        mult_array[counter].fill(i/96)
        counter+=1
    
    #below code gives an array of the cumsum with lifespan at each timestep for each server factored in
    cumsum_w_lifespan = np.reshape(np.array(x), (chunks, 28))
    dc1_cumsum = cumsum_w_lifespan[:, 0:7]
    dc2_cumsum = cumsum_w_lifespan[:, 7:14]
    dc3_cumsum = cumsum_w_lifespan[:, 14:21]
    dc4_cumsum = cumsum_w_lifespan[:, 21:28]
    dc_array = [dc1_cumsum,dc2_cumsum,dc3_cumsum,dc4_cumsum]
    ls_cumsum = []
    for i in range(4):
        giga = []
        step = 0
        for timestep in range(TIMESTEPS2):
            step=step%step_size
            modder = int(95/step_size)
            curr_chunk = int(timestep/step_size)
            modded_ts = curr_chunk%modder
            if(timestep >= 96):
                temp = dc_array[i][modded_ts:curr_chunk+1]#*(mult_array[step::step_size][::-1])
            else:
                mult_array[step-1:timestep+step_size-1:step_size]
                temp = dc_array[i][:curr_chunk+1]#*(mult_array[step:timestep+step_size:step_size][::-1])
            temp = np.sum(temp, axis=0)
            giga.append(temp)
            step+=1
        ls_cumsum.append(giga)
    #array of datacenter and custom cumsum at each timestep shape(datacenter, timestep, servergen), 
    # expired servers are auto excluded
    ls_cumsum = np.array(ls_cumsum)
    #add constraint for z where z must be less than or equal to cumsum x value at its corresponding timestep
    #(cant dismiss more servers than bought)
    # for timestep in range(TIMESTEPS2):
    #     for datacenter in range(4):
    #         dc_id = index_to_dcid[datacenter]
    #         sens_demand = demand2[demand2["datacenter_id"] == dc_id].drop_duplicates(subset="time_step")
    #         #filter for the timesteps we need
    #         sens_demand = sens_demand[sens_demand["time_step"].isin(np.arange(START_STEP2, TIMESTEPS2+START_STEP2))]
    #         sens_demand = sens_demand.drop(columns=["time_step","datacenter_id","latency_sensitivity"]).to_numpy().astype("int")
            

    for timestep in range(TIMESTEPS2):
        for datacenter in range(4):
            dc_id = index_to_dcid[datacenter]
            sens_demand = demand2[demand2["datacenter_id"] == dc_id].drop_duplicates(subset="time_step")
            #filter for the timesteps we need
            sens_demand = sens_demand[sens_demand["time_step"].isin(np.arange(START_STEP2, TIMESTEPS2+START_STEP2))]
            sens_demand = sens_demand.drop(columns=["time_step","datacenter_id","latency_sensitivity"]).to_numpy().astype("int")
            #total slots occupied cannot exceed dc capacity at any timeframe
            #if servers have started expiring
            chunk = int(timestep/step_size)
            modder = int(96/step_size)
            if(timestep>=96):
                if(timestep%step_size==0):
                    cumsum_no_expired = np.subtract(cumsum_x[chunk][datacenter], cumsum_x[chunk-modder][datacenter])
                    #total used cap cannot be more than dc_cap
                    cumsum_cpu_no_expired = np.subtract(cumsum_x[chunk][datacenter][:4], cumsum_x[chunk-modder][datacenter][:4])
                    cumsum_gpu_no_expired = np.subtract(cumsum_x[chunk][datacenter][4:], cumsum_x[chunk-modder][datacenter][4:])
                    solver.Add(np.sum(cumsum_cpu_no_expired*2)+np.sum(cumsum_gpu_no_expired*4)
                    <= dc_cap[datacenter])

                if(datacenter != 3):
                    for servergen in range(7):
                        index = timestep*21+datacenter*7+servergen
                        if(datacenter == 2):
                            #cumsum_no_expiredDC4 = np.subtract(cumsum_x[step][3], cumsum_x[step-(96/step_size)][3])
                            #supply at high should be less than sensdemand and less than total datacenter supply for high
                            solver.Add(r[index] <= sens_demand[timestep][servergen])
                            solver.Add(r[index] <= ls_cumsum[datacenter][timestep][servergen]*capacity[servergen]
                                +ls_cumsum[datacenter+1][timestep][servergen]*capacity[servergen])
                        else:
                            solver.Add(r[index] <= sens_demand[timestep][servergen])
                            solver.Add(r[index] <= ls_cumsum[datacenter][timestep][servergen]*capacity[servergen])

            #if none can/have expire yet
            else:
                #total used cap cannot be more than dc_cap
                solver.Add(np.sum(cumsum_x[chunk][datacenter][:4]*2)+np.sum(cumsum_x[chunk][datacenter][4:]*4)
                 <= dc_cap[datacenter])
                if(datacenter != 3):
                    for servergen in range(7):
                        index = timestep*21+datacenter*7+servergen
                        if(datacenter == 2):
                            #high latency demand should be less than sensdemand and less than total dc supply
                            solver.Add(r[index] <= sens_demand[timestep][servergen])
                            solver.Add(r[index] <= ls_cumsum[datacenter][timestep][servergen]*capacity[servergen]
                                +ls_cumsum[datacenter+1][timestep][servergen]*capacity[servergen])
                        else:
                            solver.Add(r[index] <= sens_demand[timestep][servergen])
                            solver.Add(r[index] <= ls_cumsum[datacenter][timestep][servergen]*capacity[servergen])

    #print("Number of constraints =", solver.NumConstraints())

    #solver.parameters.max_time_in_seconds = 10.0

    # Objective
    solver.Maximize(objective_func(demand2, x, r, step_size, TIMESTEPS2, START_STEP))
    
    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Total value =', make_readable(solver.Objective().Value()))
    else:
        print('The problem does not have an optimal solution.')

    if status == pywraplp.Solver.OPTIMAL:
        result_df = result_to_df(x,r,step_size,START_STEP2,TIMESTEPS2)
    else:
        return "no solution"
    
    return result_df

# %%
result_df = max_profit(pd.read_csv("../data/demand.csv"))
result_df.to_csv('out.csv', index=True)  
valid = verify_solution_integrity(result_df, True)
if(not valid):
    print("solution has an error!")
