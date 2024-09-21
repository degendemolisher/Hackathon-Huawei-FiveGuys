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
 'CPU.S1', 'CPU.S2', 'CPU.S3', 'CPU.S4', 'GPU.S1', 'GPU.S2', 'GPU.S3','discards']

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
def result_to_df(result_x, result_z, step_size, start_step, time_steps, solver):
    results_dict = dict_gen(time_steps)
    chunks = int(time_steps/step_size)
    x_solution = np.array([solver.Value(result_x[i]) for i in range(len(result_x))])
    x_solution = np.reshape(x_solution,(chunks, 4, 7))
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
            tot_z = []
            for servergen in range(7):
                if((i-1)%step_size==0):
                    results_dict[row_counter].append(int(x_solution[chunk, dc, servergen]))
                    count+=1
                else:
                    results_dict[row_counter].append(0)
            if(i%24==0 and (i!=0 and i!=168)):
                length = len(result_z[i])
                for number in range(0,length,28):
                    for j in result_z[i][number+dc*7:number+dc*7+7]:
                        tot_z.append(int(solver.Value(j)))
            else:
                tot_z.append(0)
            results_dict[row_counter].append(tot_z)
            row_counter += 1
        ts+=1
    result_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=columns)
    return result_df

def z_stuff(re_z1):
    re_z1_cumsum = np.cumsum(re_z1, axis=0)
    # re_z1_cumsum = np.cumsum(re_z1_cumsum, axis=1)
    # print("------")
    # print(re_z1_cumsum[0][0][0][0])
    # print(re_z1_cumsum[0][1][0][0])
    # print(re_z1_cumsum[1][0][0][0])
    # print(re_z1_cumsum[1][1][0][0])
    # print(re_z1_cumsum[1][2][0][0])
    # print(re_z1_cumsum[2][0][0][0])
    # print(re_z1_cumsum[2][1][0][0])
    # print(re_z1_cumsum[2][2][0][0])
    # print(re_z1_cumsum[2][3][0][0])
    # print("aaaa")
    # print(re_z1_cumsum[4][4][0][0])
    # print(re_z1_cumsum[5][4][0][0])
    return re_z1_cumsum

# %%
dc_cap = datacenters["slots_capacity"].to_numpy()
server_energies = servers["energy_consumption"].to_numpy()
purchase_prices = servers["purchase_price"].to_numpy()
capacity = servers["capacity"].to_numpy()

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

def profit(demand, x, z, y, cumsum_x, step_size, TIMESTEPS, START_STEP):
    chunks = int(TIMESTEPS/step_size)
    x = np.reshape(x,(chunks, 4, 7))
    y = np.reshape(y,(TIMESTEPS,3,7))
    #x = shape(TIMESTEPS,DATACENTER,SERVERGEN)
    all_lifespanner = []
    revenues = []
    costs = []
    #for each datacenter
    re_z = {}
    re_z1 = []
    re_z2 = []
    #put in values for ts 0 (all zeros)
    x_chunks_num = int(168/step_size)
    re_z2.append(np.zeros((x_chunks_num,4,7)))
    for i in z.keys():
        l = len(z[i])
        l = int(l/28)
        temmp = np.reshape(z[i],(l, 4, 7))
        #forbidden technique(filler) get how much of array "future" should be filled with 0
        future_filler = int((168-i)/step_size)
        temmp = np.append(temmp, np.zeros((future_filler,4,7)))
        if(i>=96):
            past_filler = int((i-(96-step_size))/step_size)
            temmp = np.insert(temmp, 0, np.zeros((past_filler*4*7)))
            temmp = np.reshape(temmp,(l+future_filler+past_filler,4,7))
        else:
            temmp = np.reshape(temmp,(l+future_filler,4,7))
        re_z[i] = temmp
        re_z1.append(temmp)
        re_z2.append(temmp)
    #array where 1st index is discard step(eg24 or 48etc) and 2nd index is the timesteps
    cumsum_z1 = z_stuff(re_z1)
    # print(cumsum_z1.shape)
    # print(cumsum_z1[0][0][0][0])
    # print(cumsum_z1[0][0][1][0])
    # print(cumsum_z1[0][0][2][0])
    # print("bbbb")
    # print(cumsum_z1[1][0][0][0])
    # print(cumsum_z1[1][0][1][0])
    # print(cumsum_z1[1][0][2][0])
    #need an array such that for all timesteps, all datacenters, all servergens, it contains an array of server-discard at that timestep
    #max number of servers bought at different timesteps in existance
    re_z2_cumsum = z_stuff(re_z2)
    max_in_existance = int(96/step_size)
    x_minus_z_arr = np.zeros((168, 4, 7, max_in_existance), dtype=object)
    # x_minus_z_arr = []
    for chunk in range(chunks):
        for datacenter in range(4):
            for servergen in range(7):
                ts_x = x[chunk, datacenter, servergen]
                start_timestep = chunk*step_size
                for timestep in range(start_timestep,min(96+start_timestep,168)):
                    disc_chunk = max(min(int(timestep/24),6),0)
                    curr_chunk = int(timestep/step_size)
                    # if(chunk>=int(144/step_size)):
                    #     adder = int((timestep-(96-step_size))/step_size)
                    #     x_minus_z_arr[timestep][datacenter][servergen][chunk-adder] = ts_x
                    #     continue
                    discard = re_z2_cumsum[disc_chunk][chunk][datacenter][servergen]
                    if(timestep>=96):
                        adder = int((timestep-(96-step_size))/step_size)
                        x_minus_z_arr[timestep][datacenter][servergen][chunk-adder] = ts_x - discard
                    else:
                        x_minus_z_arr[timestep][datacenter][servergen][chunk] = ts_x - discard

    # print(x_minus_z_arr[0][0][0][0])
    # print(x_minus_z_arr[0][0][1][0])
    #gets sum of servergens for each datacenter and timeframe NEED TO SUM THIS ASWELL FOR FINAL CALC (TOTAL SERVERGENS)
    # print(np.sum(x_minus_z_arr, axis=2)[24][0][0])
    # print(np.sum(x_minus_z_arr, axis=2)[24][0][1])
    # print(np.sum(x_minus_z_arr, axis=2).shape)

    # mult_array = np.empty((96,7))
    # for i in range(7):
    #     mult_array[:,i] = np.arange(1,97)/96

    mult_array = np.arange(1,97,step_size)

    for datacenter in range(4):
        #get generated revenue at each timestep
        dc_id = index_to_dcid[datacenter]
        lat_sens = demand[demand["datacenter_id"] == dc_id]["latency_sensitivity"].iloc[0]
        dc_selling_prices = selling_prices[selling_prices["latency_sensitivity"] == lat_sens]["selling_price"].to_numpy()
        dc_revenues = []
        max_servers = int(96/step_size)

        dc_lifespanner = []
        #skip dc4 as calc for dc3 already includes the info for dc4
        if(datacenter != 3):
            for timestep in range(TIMESTEPS):
                ts_servers = x_minus_z_arr[timestep, datacenter].copy()
                chunk = int(timestep/step_size)
                curr_max = min(max_in_existance, chunk+1)
                curr_max2 = min(max_in_existance-1, chunk)
                if(timestep>=step_size):
                    calcs = []
                    for servergen in range(7):
                        calc = dc_selling_prices[servergen] * curr_max #timestep-i*step_size
                        servers_lifespan = []
                        for i in range(curr_max2):
                            servers_lifespan.append(ts_servers[servergen][i] * mult_array[curr_max2-i])
                        calc = calc * np.sum(servers_lifespan)-(np.sum(servers_lifespan)*curr_max)
                        calcs.append(calc)
                    dc_lifespanner.append(np.sum(calcs))
                supply = y[timestep, datacenter]
                revenue = []
                for j in range(7):
                    revenue.append(supply[j]*dc_selling_prices[j])
                revenue = np.sum(revenue)
                dc_revenues.append(revenue)
            all_lifespanner.append(dc_lifespanner)
            revenues.append(dc_revenues)
        #calc energycost for all servergens at the datacenter
        energy_costs = server_energies * datacenters[datacenters["datacenter_id"] == dc_id]["cost_of_energy"].to_numpy()

        timestep_costs = []
        #starts at 2 as maintained servers start from 2life
        #REMINDER:can do buy action every step_size steps
        #get cost equation at each timestep
        chunks = int(TIMESTEPS/step_size)
        for timestep in range(TIMESTEPS):
            ts_servers = x_minus_z_arr[timestep, datacenter].copy()
            step = timestep%step_size
            chunk = int(timestep/step_size)
            new_servers = min(chunk, max_servers-1)
            #test if just bought some servers
            if(step==0):
                #get cost of new servers
                new_cost = ts_servers[:, new_servers]*(purchase_prices+energy_costs+maintenance_cost_array[0])
                #factor in lifespan
                # new_cost *=  1/96
                new_cost = np.sum(new_cost)

                #calc energy + maintenance cost
                energy_and_maint = maintenance_cost_array + energy_costs
                #factor in lifespan into energy_and_maint cost
                # energy_and_maint = energy_and_maint * mult_array
                maintained_servers = ts_servers[:, :new_servers]
                maint_cost = []

                if(timestep<96):
                    for i in range(min(chunk,max_servers-1)):
                        # print(ts_servers.shape)
                        # print(i)
                        maint_cost.append(maintained_servers[:,i] * energy_and_maint[timestep-i*step_size])
                else:
                    for i in range(max_servers-1):
                        maint_cost.append(maintained_servers[:,i] * energy_and_maint[(95-step_size+step)-i*step_size])

                # for i in range(min(chunk,7)):
                #     maint_cost.append(maintained_servers[:,i] * energy_and_maint[i*step_size+step])
                # print(np.array(maint_cost).shape)
                maint_cost = np.sum(maint_cost)
                timestep_costs.append(maint_cost + new_cost)

            #no servers bought this timestep
            else:
                maintained_servers = ts_servers
                #calc energy + maintenance cost
                energy_and_maint = maintenance_cost_array + energy_costs
                # energy_and_maint = energy_and_maint * mult_array
                maint_cost = []

                if(timestep<96):
                    for i in range(min(chunk+1,max_servers)):
                        maint_cost.append(maintained_servers[:,i] * energy_and_maint[timestep-i*step_size])
                else:
                    for i in range(max_servers):
                        maint_cost.append(maintained_servers[:,i] * energy_and_maint[(95-step_size+step)-i*step_size])

                # for i in range(min(chunk+1,8)):
                #     maint_cost.append(maintained_servers[:,i] * energy_and_maint[i*step_size+step])
                maint_cost = np.sum(maint_cost)
                timestep_costs.append(maint_cost)
        costs.append(timestep_costs)
        

    #after all of the profits and costs have been calculated for all the datacenters at each timestep,
    #get sum of costs for the datacenters and the sum of profits for all datacenters at each timestep
    lifespan_sum = np.sum(all_lifespanner, axis=0)
    costs_sum = np.sum(costs, axis=0)
    revenue_sum = np.sum(revenues, axis=0)
    return revenue_sum, costs_sum, lifespan_sum

def objective_func(demand, x, z, y, cumsum_x, step_size, TIMESTEPS, START_STEP):
    revenues, costs, lifespan = profit(demand, x, z, y, cumsum_x, step_size, TIMESTEPS, START_STEP)
    #get profit at each timestep
    profit_arr = []
    for i in range(TIMESTEPS):
        # if(i>=step_size):
        #     ts_profit = (lifespan[i-step_size]+(revenues[i]-costs[i]))*(1/96)
        # else:
        #     ts_profit = (revenues[i]-costs[i])*(i/96)
        ts_profit = revenues[i]-costs[i]
        profit_arr.append(ts_profit)
    Objective = np.sum(profit_arr)
    return Objective

# %%
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model

def max_profit(demand, elasticity=[], prices=[], prices_step_size=12, step_size=6, START_STEP=1, TIMESTEPS=168):
    demand2 = demand.merge(datacenters, on="latency_sensitivity", how="left")
    START_STEP2 = START_STEP
    TIMESTEPS2 = TIMESTEPS
    START_STEP %= 96
    TIMESTEPS %= 96
    chunks = int(TIMESTEPS2/step_size)
    # Create the solver
    # solver = pywraplp.Solver.CreateSolver("SAT")
    solver = cp_model.CpModel()

    new_demand = []
    delta_p = []
    for latency in range(3):
        for timestep in range(TIMESTEPS2):
            for servergen in range(7):
                dp = (prices[timestep, latency, servergen]-purchase_prices[servergen])/purchase_prices[servergen]
                delta_p.append(dp)

    #SELF NOTE: PRICES WILL BE AN ARRAY OF SHAPE:TIMESTEPS,LATENCY,SERVERGEN (same as y) CAN HAVE STEP_SIZE ASWELL

    # Variables
    # x is the bought servergens at each timestep chunk/checkpoint
    x = []
    # z is the number of servers to dismiss, shape is such that it is called every n steps and at that 
    # step there should be enough z variables to dismiss from any of the servers from before that havent expired
    # This gives z a weirdo shape but the overall idea for its shape should be number of times it is "called" 
    # (its chunks),* every non-expired x variable up to that chunk(exclusive). 
    z = {}
    z_arr = []
    #array that represents y=min(x*cap,demand) at each timestep
    y = []
    #boolean variable that each bind to a x val, used to add if then links later NEEDS SIMILAR GOOFY SHAPE AS Z
    b = []
    #array of profit chain for the boolean to apply lifespan to profit calcs
    profit_chain = []
    c = 0
    #makes an array of size (chunks * dc_num * servergen_num)
    for i in range(chunks):
        #for all 4 datacenters
        for k in range(4):
            a=0
            #generate cpu servers
            for j in range(4):
                x.append(solver.NewIntVar(0, int(dc_cap[k]/2), f'x{c}'))
                # b.append(solver.NewBoolVar(f"b{c}"))
                a+=1
                c+=1
            #generate gpu servers
            for j in range(3):
                x.append(solver.NewIntVar(0, int(dc_cap[k]/4), f'x{c}'))
                # b.append(solver.NewBoolVar(f"b{c}"))
                a+=1
                c+=1
    
    c=0
    for i in range(TIMESTEPS2):
        for k in range(3):
            a=0
            #generate cpu servers
            for j in range(4):
                y.append(solver.NewIntVar(0, int(dc_cap[k]/2)*capacity[a], f'y{c}'))
                a+=1
                c+=1
            #generate gpu servers
            for j in range(3):
                y.append(solver.NewIntVar(0, int(dc_cap[k]/4)*capacity[a], f'y{c}'))
                a+=1
                c+=1

    #get cumulative sum of the servergen at each timesteps
    cumsum_x = np.reshape(np.array(x), (chunks, 28))
    cumsum_x = np.cumsum(cumsum_x, axis=0)
    cumsum_x = np.reshape(cumsum_x, (chunks, 4, 7))

    temp_x = np.reshape(np.array(x), (chunks, 4, 7))

    dismiss_steps = 24
    #0,24,48,72,96,120,144,168
    #-1 as cannot dismiss on step 0, now with step-24 gives 6
    c=0
    dismiss_chunks = int(TIMESTEPS2/dismiss_steps)-1
    #collect is for later
    collect = {}
    # ts_affector_full = []
    # ts_affector = []
    # ts_counter = 0
    for x_chunk in range(chunks):
        timestep = x_chunk*step_size
        tempz = []
        if(timestep >= 96):
            exlude_chunk = int((x_chunk*step_size-96)/step_size)+1
            x_number = temp_x[exlude_chunk:x_chunk]
        else:
            x_number = temp_x[:x_chunk]
        if(timestep%dismiss_steps==0 and timestep!=0):
            # if(ts_counter>4):
            #     ts_affector_full.append(ts_affector[0])
            #     ts_affector.pop(0)
            for window in range(x_number.shape[0]):
                for dc in range(x_number.shape[1]):
                    for s_gen in range(x_number.shape[2]):
                        if(s_gen<4):
                            temp_z = solver.NewIntVar(0, int(dc_cap[k]/2), f't{timestep}z{c}dc{dc}')
                            tempz.append(temp_z)
                        else:
                            temp_z = solver.NewIntVar(0, int(dc_cap[k]/4), f't{timestep}z{c}dc{dc}')
                            tempz.append(temp_z)
                        c+=1
            # structure of z is a dict that contains the timestep value at the steps of z and an array corresponding
            # to the z val matched to a value in x such that it matches 1:1 if x was flattened and is of size:
            # (buy_stepsdone* datacenter* servergen) for each time z is called
            # made such that z is matched with x without expired servers (theyre auto excluded)
            # with dimiss_step = 24 sizes are: 56, 112, 168, 196, 196, 196
            z[timestep] = np.array(tempz)
            collect[timestep] = x_number.flatten()
                

    # print("Number of variables =", solver.NumVariables())

    #for each datacenter
    re_z = {}
    re_z1 = []
    re_z2 = []
    x_chunks_num = int(168/step_size)
    #put in values for ts 0 (all zeros)
    re_z2.append(np.zeros((x_chunks_num,4,7)))
    for i in z.keys():
        l = len(z[i])
        #datacenter_number*server_gen_number = 28
        l = int(l/28)
        temmp = np.reshape(z[i],(l, 4, 7))
        #forbidden technique(filler) get how much of array "future" should be filled with 0
        future_filler = int((168-i)/step_size)
        temmp = np.append(temmp, np.zeros((future_filler,4,7)))
        if(i>=96):
            past_filler = int((i-(96-step_size))/step_size)
            temmp = np.insert(temmp, 0, np.zeros((past_filler*4*7)))
            temmp = np.reshape(temmp,(l+future_filler+past_filler,4,7))
        else:
            temmp = np.reshape(temmp,(l+future_filler,4,7))
        re_z[i] = temmp
        re_z1.append(temmp)
        re_z2.append(temmp)
    re_z1 = np.array(re_z1)
    re_z2 = np.array(re_z2)
    re_z1_cumsum = np.cumsum(re_z1, axis=0)
    re_z2_cumsum = np.cumsum(re_z2, axis=0)

    #get x-z at each timestep for every server that exists at that timestep
    max_in_existance = int(96/step_size)
    x_minus_z_arr = np.zeros((168, 4, 7, max_in_existance), dtype=object)
    # x_minus_z_arr = []
    discard_chunks = int(TIMESTEPS2/dismiss_steps)
    for chunk in range(chunks):
        for datacenter in range(4):
            for servergen in range(7):
                ts_x = temp_x[chunk][datacenter][servergen]
                start_timestep = chunk*step_size
                for timestep in range(start_timestep,min(96+start_timestep,168)):
                    disc_chunk = max(min(int(timestep/24),6),0)
                    curr_chunk = int(timestep/step_size)
                    # if(chunk>=int(144/step_size)):
                    #     adder = int((timestep-(96-step_size))/step_size)
                    #     x_minus_z_arr[timestep][datacenter][servergen][chunk-adder] = ts_x
                    #     continue
                    discard = re_z2_cumsum[disc_chunk][chunk][datacenter][servergen]
                    if(timestep>=96):
                        adder = int((timestep-(96-step_size))/step_size)
                        x_minus_z_arr[timestep][datacenter][servergen][chunk-adder] = ts_x - discard
                    else:
                        x_minus_z_arr[timestep][datacenter][servergen][chunk] = ts_x - discard
    
    #init the b binding variables
    # for chunk in range(chunks):
    #     curr_max = min(max_in_existance, chunk+1)
    #     for datacenter in range(4):
    #         for servergen in range(7):
    #             for server in range(curr_max):
    #                 b.append(solver.NewBoolVar(f"b{c}"))


    #below code gives an array of the cumsum with lifespan at each timestep for each server factored in
    cumsum_w_lifespan = np.reshape(np.array(x), (chunks, 28))
    dc1_cumsum = cumsum_w_lifespan[:, 0:7]
    dc2_cumsum = cumsum_w_lifespan[:, 7:14]
    dc3_cumsum = cumsum_w_lifespan[:, 14:21]
    dc4_cumsum = cumsum_w_lifespan[:, 21:28]
    dc_array = [dc1_cumsum,dc2_cumsum,dc3_cumsum,dc4_cumsum]
    ls_cumsum = []
    for dc in range(4):
        giga = []
        step = 0
        for timestep in range(TIMESTEPS2):
            step=step%step_size
            modder = int(95/step_size)
            curr_chunk = int(timestep/step_size)
            modded_ts = curr_chunk%modder
            if(timestep >= 96):
                temp = dc_array[dc][modded_ts:curr_chunk+1]#*(mult_array[step::step_size][::-1])
            else:
                # mult_array[step-1:timestep+step_size-1:step_size]
                temp = dc_array[dc][:curr_chunk+1]#*(mult_array[step:timestep+step_size:step_size][::-1])
            temp = np.sum(temp, axis=0)
            giga.append(temp)
            step+=1
        ls_cumsum.append(giga)
    #array of datacenter and custom cumsum at each timestep shape(datacenter, timestep, servergen), 
    # expired servers are auto excluded
    ls_cumsum = np.array(ls_cumsum)
    # print(cumsum_z[2][0])
    # print("ls",ls_cumsum[0][100][0])

    # Constraints
    #adds constraint for retail time
    for dc in range(4):
        action_time_array = np.arange(1,TIMESTEPS2+1,step_size)
        start_pos = dc*7
        for servergen in range(7):
            rt = eval(release_times[servergen])
            counter = servergen
            for j in action_time_array:
                if(j < rt[0] or j > rt[1]):
                    solver.Add(x[start_pos+counter] == 0)
                counter+=28

    # mult_array = np.empty(((96),7))
    # counter=0
    # for i in range(1,96+1):
    #     mult_array[counter].fill(i/96)
    #     counter+=1

    for timestep in range(TIMESTEPS2):
        disc_chunk = min(int(timestep/24)-1,5)
        no_expired = max(int((timestep-(96-step_size))/step_size),0)
        chunk = int(timestep/step_size)
        modder = int(96/step_size)
        if(timestep%step_size==0):
            for datacenter in range(4):
                #datacenter cap constraint
                cpu = x_minus_z_arr[timestep][datacenter][:4]*2
                gpu = x_minus_z_arr[timestep][datacenter][4:]*4
                solver.Add(np.sum(cpu)+np.sum(gpu) <= dc_cap[datacenter])
                solver.Add(np.sum(cpu)+np.sum(gpu) >= 0)


    #to ensure 1:1 mapping we must also add the restiction of: x - all z affecting that timestep >=0
    #e.g. x0 - z0-z56-z172 >=0
    # print(re_z1_cumsum[5][0][0][0])
    for datacenter in range(4):
        dc_id = index_to_dcid[datacenter]
        sens_demand = demand2[demand2["datacenter_id"] == dc_id].drop_duplicates(subset="time_step")
        #filter for the timesteps we need
        sens_demand = sens_demand[sens_demand["time_step"].isin(np.arange(START_STEP2, TIMESTEPS2+START_STEP2))]
        sens_demand = sens_demand.drop(columns=["time_step","datacenter_id","latency_sensitivity","cost_of_energy","slots_capacity"]).to_numpy().astype("int")
        sd_max = np.reshape(sens_demand,(chunks,step_size,7))
        sd_max = np.max(sd_max, axis=1)
        for timestep in range(TIMESTEPS2):
            disc_chunk = min(int(timestep/24)-1,5)
            no_expired = max(int((timestep-(96-step_size))/step_size),0)
            chunk = int(timestep/step_size)
            if(timestep>=24):
                if(datacenter != 3):
                    discards_dc4 = np.sum(re_z1_cumsum[disc_chunk,no_expired:chunk,3],axis=0)
                    discards = np.sum(re_z1_cumsum[disc_chunk,no_expired:chunk,datacenter],axis=0)
                    for servergen in range(7):
                        index = timestep*21+datacenter*7+servergen
                        if(datacenter == 2):
                            #demand met constraint
                            x_minus_z_dc3 = ls_cumsum[datacenter][timestep][servergen]-discards[servergen]
                            x_minus_z_dc4 = ls_cumsum[datacenter+1][timestep][servergen]-discards_dc4[servergen]
                            solver.Add(y[index] <= x_minus_z_dc3*capacity[servergen]+x_minus_z_dc4*capacity[servergen])
                            solver.Add(y[index] <= sens_demand[timestep][servergen])
                        else:
                            #demand met constraint
                            x_minus_z = ls_cumsum[datacenter][timestep][servergen]-discards[servergen]
                            #dc(xa-za)*cap<=demand
                            solver.Add(y[index] <= x_minus_z*capacity[servergen])
                            solver.Add(y[index] <= sens_demand[timestep][servergen])
            else:
                if(datacenter != 3):
                    for servergen in range(7):
                        index = timestep*21+datacenter*7+servergen
                        if(datacenter == 2):
                            #demand met constraint
                            x_dc3 = ls_cumsum[datacenter][timestep][servergen]
                            x_dc4 = ls_cumsum[datacenter+1][timestep][servergen]
                            solver.Add(y[index] <= x_dc3*capacity[servergen]+x_dc4*capacity[servergen])
                            solver.Add(y[index] <= sens_demand[timestep][servergen])
                        else:
                            #demand met constraint
                            solver.Add(y[index] <= ls_cumsum[datacenter][timestep][servergen]*capacity[servergen] )
                            solver.Add(y[index] <= sens_demand[timestep][servergen])


            # if(timestep%step_size==0):
            #     if(timestep>=24):
            #         #datacenter cap constraint
            #         discards = np.sum(re_z1_cumsum[disc_chunk,no_expired:chunk,datacenter],axis=0)
            #         x_minus_z = ls_cumsum[datacenter][timestep]-discards
            #         cpu_cap_used = x_minus_z[:4]*2
            #         gpu_cap_used = x_minus_z[4:]*4
            #         solver.Add(np.sum(cpu_cap_used)+np.sum(gpu_cap_used) <= dc_cap[datacenter])
            #         solver.Add(np.sum(cpu_cap_used)+np.sum(gpu_cap_used) >= 0)
            #     else:
            #         #datacenter cap constraint
            #         cpu_cap_used = ls_cumsum[datacenter][timestep][:4]*2
            #         gpu_cap_used = ls_cumsum[datacenter][timestep][4:]*4
            #         # print(np.sum(ls_cumsum[datacenter][timestep]))
            #         solver.Add(np.sum(cpu_cap_used)+np.sum(gpu_cap_used) <= dc_cap[datacenter])
            #         solver.Add(np.sum(cpu_cap_used)+np.sum(gpu_cap_used) >= 0)
                for servergen in range(7):
                    x_minus_z_1to1 = temp_x[chunk][datacenter][servergen] - re_z1_cumsum[5][chunk][datacenter][servergen]
                    # print(x_minus_z_1to1)
                    solver.Add(x_minus_z_1to1 >= 0)
            curr_max = min(max_in_existance, chunk+1)
            for servergen in range(7):
                for server in range(curr_max):
                    s = x_minus_z_arr[timestep][datacenter][servergen][server]
                    solver.Add(s >= 0)

    
    # prev_indexes = 0
    # for chunk in range(chunks):
    #     curr_max = min(max_in_existance, chunk+1)
    #     # rough = np.arange(1,curr_max)*rough
    #     for datacenter in range(4):
    #         for servergen in range(7):
    #             for server in range(curr_max):
    #                 index = prev_indexes + datacenter*7*curr_max + servergen*curr_max+server
    #                 # print(index)
    #                 s = x_minus_z_arr[chunk][datacenter][servergen][server]
    #                 solver.Add(s > 0).only_enforce_if(b[index])
    #                 solver.Add(s == 0).only_enforce_if(~b[index])
    #     prev_indexes += datacenter*7*curr_max + servergen*curr_max+server
    
    # profit_arr, cost_arr = profit(demand2, x, z, y, ls_cumsum, step_size, TIMESTEPS2, START_STEP)
    # for timestep in range(TIMESTEPS2):
    #     curr_chunk = int(timestep/step_size)
    #     for datacenter in range(4):
    #         for servergen in range(7):


    #print("Number of constraints =", solver.NumConstraints())

    #solver.parameters.max_time_in_seconds = 10.0

    # Objective
    solver.Maximize(objective_func(demand2, x, z, y, ls_cumsum, step_size, TIMESTEPS2, START_STEP))
    
    # Solve
    solver2 = cp_model.CpSolver()
    status = solver2.Solve(solver)
    # status = solver.Solve()
    if status == cp_model.OPTIMAL:
        print('Total value =', make_readable(solver2.ObjectiveValue()))
    else:
        print('The problem does not have an optimal solution.')

    if status == cp_model.OPTIMAL:
        result_df = result_to_df(x,z,step_size,START_STEP2,TIMESTEPS2, solver2)
    else:
        return "no solution"
    
    return result_df

# %%
# result_df = max_profit(pd.read_csv("../data/demand.csv"))
# print(result_df)
# result_df.to_csv('out2.csv', index=True)
# valid = verify_solution_integrity(result_df, True)
# if(not valid):
#     print("solution has an error!")
