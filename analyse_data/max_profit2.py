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
def result_to_df(result_x, result_z, step_size, start_step, time_steps):
    results_dict = dict_gen(time_steps)
    chunks = int(time_steps/step_size)
    result_x = np.reshape(result_x,(chunks, 4, 7))
    #result_z = np.reshape(rresult_z,(6,4,7))
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
                    results_dict[row_counter].append(int(result_x[chunk, dc, servergen].solution_value()))
                    count+=1
                else:
                    results_dict[row_counter].append(0)
            if(i%24==0 and (i!=0 and i!=168)):
                for j in result_z[i]:
                    tot_z.append(int(j.solution_value()))
            else:
                tot_z.append(0)
            results_dict[row_counter].append(tot_z)
            row_counter += 1
        ts+=1
    result_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=columns)
    return result_df

def z_stuff(re_z1):
    # ts_chunks = []
    # for i in range(1,len(re_z1)+1):
    #     mmm = int((i*24-84)/12)
    #     mmm2 = max(mmm,0)
    #     print((re_z1[:i][mmm2:i*2][dc]).shape)
    #     sss = np.sum(re_z1[:i][mmm2:i*2][dc],axis=0)
    #     sss2 = np.sum(re_z1[:i][:i*2][dc], axis=0)
    #     if(i==3):
    #         print("sss",sss[0][0])
    #     #print(sss)
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

def profit(demand, x, z, cumsum_x, step_size, TIMESTEPS, START_STEP):
    chunk = TIMESTEPS/step_size
    #x and y = shape(TIMESTEPS,DATACENTER,SERVERGEN)
    #get cumulative sum of number of servers for all servergens
    revenues = []
    costs = []
    #for each datacenter
    re_z = {}
    re_z1 = []
    for i in z.keys():
        l = len(z[i])
        l = int(l/28)
        temmp = np.reshape(z[i],(l, 4, 7))
        #forbidden technique(filler) get how much of array "future" should be filled with 0
        future_filler = int((144-i)/step_size)
        temmp = np.append(temmp, np.zeros((future_filler,4,7)))
        if(i>=96):
            past_filler = int((i-84)/12)
            temmp = np.insert(temmp, 0, np.zeros((past_filler*4*7)))
            temmp = np.reshape(temmp,(l+future_filler+past_filler,4,7))
        else:
            temmp = np.reshape(temmp,(l+future_filler,4,7))
        re_z[i] = temmp
        re_z1.append(temmp)
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

    for datacenter in range(4):
        #get generated revenue at each timestep
        dc_id = index_to_dcid[datacenter]
        lat_sens = demand[demand["datacenter_id"] == dc_id]["latency_sensitivity"].iloc[0]
        dc_selling_prices = selling_prices[selling_prices["latency_sensitivity"] == lat_sens]["selling_price"].to_numpy()
        dc_revenues = []

        mult_array = np.empty((95,7))
        for i in range(7):
            mult_array[:,i] = np.arange(2,97)/96
        #mult_array = np.arange(1,97)/96
        
        #skip dc4 as calc for dc3 already includes the info for dc3
        if(datacenter != 3):
            for timestep in range(TIMESTEPS):
                max_servergens = int(96/step_size)
                curr_chunk = int(timestep/step_size)+1
                dc = datacenter
                buy_chunk = min(int(timestep/step_size)-1,11)
                disc_chunk = min(int(timestep/24)-1,5)
                no_expired = max(int((timestep-84)/step_size),0)
                if(timestep>=24):
                    discards = np.sum(cumsum_z1[disc_chunk,no_expired:,dc], axis=0)
                    discards_dc3 = np.sum(cumsum_z1[disc_chunk,no_expired:,3], axis=0)
                    if(datacenter == 2):
                        total_ts_x = cumsum_x[datacenter][timestep]+cumsum_x[datacenter+1][timestep]
                        total_ts_x = total_ts_x-discards-discards_dc3
                    else:
                        # if(timestep%12==0):
                        #     print("d: ",discards[0])
                        total_ts_x = cumsum_x[datacenter][timestep]
                        total_ts_x = total_ts_x-discards
                        # print(timestep)
                else:
                    if(datacenter == 2):
                        total_ts_x = cumsum_x[datacenter][timestep]+cumsum_x[datacenter+1][timestep]
                    else:
                        total_ts_x = cumsum_x[datacenter][timestep]

                total_ts_x = total_ts_x*capacity / min(curr_chunk,max_servergens)
                revenue = 0
                # print(len(total_ts_x))
                # if(len(total_ts_x)==4):
                #     print(total_ts_x[0][0])
                #     print(total_ts_x[0][1])
                #     print(len(total_ts_x[0]))
                #     print(total_ts_x[1][0])
                #     print(total_ts_x[1][1])
                for j in range(7):
                    revenue += total_ts_x[j]*dc_selling_prices[j]
                dc_revenues.append(revenue)
            revenues.append(dc_revenues)
        #calc energycost for all servergens at the datacenter
        energy_costs = server_energies * datacenters[datacenters["datacenter_id"] == dc_id]["cost_of_energy"].to_numpy()

        timestep_costs = []
        #starts at 2 as maintained servers start from 2life
        #REMINDER:can do buy action every step_size steps
        #get cost equation at each timestep
        chunks = int(TIMESTEPS/step_size)
        timestep = 0
        for i in range(chunks):
            for step in range(step_size):
                #xshape=(chunks,dc,servergen)
                #get servers that have been maintained (not new) for that datacenter
                #calc cost of the new servers and add to overall cost at end
                #if buy action was performable on this timestep:
                max_servergens=int(96/step_size)
                curr_chunk = int(i/step_size)+1
                if(step==0):
                    #after a certain timeframe servers will have started to expire
                    if(timestep>=96):
                        maintained_servers = x[i%7:i, datacenter].copy()
                    else:
                        maintained_servers = x[:i, datacenter].copy()
                    disc_chunk = min(int((timestep+1)/24)-1,5)
                    no_expired = max(int((timestep+1-84)/step_size),0)
                    if(timestep+1>=24):
                        discards = cumsum_z1[disc_chunk,no_expired:i,dc]
                        if(timestep>=156):
                            maintained_servers[:6] = maintained_servers[:6]-discards
                        elif(timestep>=144):
                            maintained_servers[:i-1] = maintained_servers[:i-1]-discards
                        elif(i%2==0):
                            maintained_servers = maintained_servers-discards
                        else:
                            print("d: ",discards[0][0])
                            print("m: ",maintained_servers[0])
                            print(timestep)
                            maintained_servers = maintained_servers-discards
                            print("m2: ", maintained_servers[0][0])
                    new_cost = x[i, datacenter] * np.rint((purchase_prices + energy_costs + maintenance_cost_array[0])*1/96 / min(curr_chunk,max_servergens)).astype("int")
                    new_cost = np.sum(new_cost)
                    #calc energy + maintenance cost
                    energy_and_maint = maintenance_cost_array[1:timestep+1] + energy_costs
                    energy_and_maint = np.rint(energy_and_maint).astype("int")
                    
                    #multiply corresponding servers with their cost to get total for servergen at each ts
                    if(timestep>=96):
                        maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep-1:step_size][:7][::-1])
                        maint_cost = np.multiply(maint_cost, mult_array[step:timestep:step_size][:7][::-1])/ min(curr_chunk,max_servergens)
                        maint_cost = np.sum(maint_cost)
                    else:
                        maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep-1:step_size][::-1])
                        maint_cost = np.multiply(maint_cost, mult_array[step:timestep:step_size][::-1])/ min(curr_chunk,max_servergens)
                        maint_cost = np.sum(maint_cost)

                    if(maint_cost == 0):
                        timestep_costs.append(new_cost)
                    else:
                        timestep_costs.append(maint_cost + new_cost)
                else:
                    if(timestep>=96):
                        maintained_servers = x[i%7:i+1, datacenter].copy()
                    else:
                        maintained_servers = x[:i+1, datacenter].copy()
                    disc_chunk = min(int((timestep+1)/24)-1,5)
                    no_expired = max(int((timestep+1-84)/step_size),0)
                    if(timestep+1>=24):
                        # print(discards)
                        # print(maintained_servers)
                        # print(maintained_servers.shape)
                        discards = cumsum_z1[disc_chunk,no_expired:i,dc]
                        maintained_servers[:len(discards)] = maintained_servers[:len(discards)]-discards
                    #calc energy + maintenance cost
                    energy_and_maint = maintenance_cost_array[:timestep+1] + energy_costs
                    energy_and_maint = np.rint(energy_and_maint).astype("int")
                    #multiply corresponding servers with their cost to get total for servergen at each ts
                    maint_cost = np.multiply(maintained_servers, energy_and_maint[step:timestep+step_size:step_size][::-1])
                    maint_cost = np.multiply(maint_cost, mult_array[step-1:timestep+step_size-1:step_size][::-1]) / min(curr_chunk,max_servergens)
                    #(maint_cost)
                    maint_cost = np.sum(maint_cost)
                    # if(timestep<14 and datacenter==0):
                    #     print("m:",maint_cost)
                    timestep_costs.append(maint_cost)
                timestep+=1
        costs.append(timestep_costs)
        

    #after all of the profits and costs have been calculated for all the datacenters at each timestep,
    #get sum of costs for the datacenters and the sum of profits for all datacenters at each timestep
    # print(costs[0][48])
    costs_sum = np.sum(costs, axis=0)
    revenue_sum = np.sum(revenues, axis=0)

    profit_arr = []
    #get profit at each timestep
    for i in range(TIMESTEPS):
        profit_arr.append(revenue_sum[i]-costs_sum[i])
    return profit_arr

def objective_func(demand, x, z, cumsum_x, step_size, TIMESTEPS, START_STEP):
    chunk = int(TIMESTEPS/step_size)
    x = np.reshape(x,(chunk, 4, 7))
    P = profit(demand, x, z, cumsum_x, step_size, TIMESTEPS, START_STEP)
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
    # z is the number of servers to dismiss, shape is such that it is called every n steps and at that 
    # step there should be enough z variables to dismiss from any of the servers from before that havent expired
    # This gives z a weirdo shape but the overall idea for its shape should be number of times it is "called" 
    # (its chunks),* every non-expired x variable up to that chunk(exclusive). 
    z = {}
    z_arr = []
    c = 0
    #makes an array of size (chunks * dc_num * servergen_num)
    for i in range(chunks):
        #for all 4 datacenters
        for k in range(4):
            #generate cpu servers
            for j in range(4):
                x.append(solver.IntVar(0, int(dc_cap[k]/2), f'x{c}'))
                c+=1
            #generate gpu servers
            for j in range(3):
                x.append(solver.IntVar(0, int(dc_cap[k]/4), f'x{c}'))
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
                        if(s_gen<=3):
                            temp_z = solver.IntVar(0, int(dc_cap[k]/2), f't{timestep}z{c}dc{dc}')
                            tempz.append(temp_z)
                        else:
                            temp_z = solver.IntVar(0, int(dc_cap[k]/4), f't{timestep}z{c}dc{dc}')
                            tempz.append(temp_z)
                        c+=1
            # structure of z is a dict that contains the timestep value at the steps of z and an array corresponding
            # to the z val matched to a value in x such that it matches 1:1 if x was flattened and is of size:
            # (buy_stepsdone* datacenter* servergen) for each time z is called
            # made such that z is matched with x without expired servers (theyre auto excluded)
            # with dimiss_step = 24 sizes are: 56, 112, 168, 196, 196, 196
            z[timestep] = np.array(tempz)
            collect[timestep] = x_number.flatten()
                

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

    mult_array = np.empty(((96),7))
    counter=0
    for i in range(1,96+1):
        mult_array[counter].fill(i/96)
        counter+=1

    #expired are auto excluded
    x_minus_z_dict = {}
    for z_chunk_ts in z.keys():
        x_minus_z_dict[z_chunk_ts] = collect[z_chunk_ts] - z[z_chunk_ts]
    # print(x_minus_z_dict[96][0])

    sum1 =  z[24][0:28]    +z[48][0:28]    +z[72][0:28] #ts0
    sum2 =  z[24][28:56]   +z[48][28:56]   +z[72][28:56] +z[96][0:28] #ts12
    sum3 =  z[48][56:84]   +z[72][56:84]   +z[96][28:56]              #ts24
    sum4 =  z[48][84:112]  +z[72][84:112]  +z[96][56:84] +z[120][0:28]#ts36
    sum5 =  z[72][112:140] +z[96][84:112]  +z[120][28:56]             #ts48
    sum6 =  z[72][140:168] +z[96][112:140] +z[120][56:84]+z[144][0:28]#ts60
    sum7 =  z[96][140:168] +z[120][84:112] +z[144][28:56] #ts72
    sum8 =  z[96][168:196] +z[120][112:140]+z[144][56:84] #ts84
    sum9 =  z[120][140:168]+z[144][84:112] #ts96
    sum10 = z[120][168:196]+z[144][112:140] #ts108
    sum11 = z[144][140:168] #ts120
    sum12 = z[144][168:196] #ts132
    cumsum_z = [sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12]
    # cumsum_z = []
    # z_range = int(TIMESTEPS2/dismiss_steps)-3
    # group_size = int((96-dismiss_steps)/dismiss_steps)
    # #get number of groups and do for all of them
    # for i in range(z_range):
    #     # first chunk them into groups where there are no expired x related ones,
    #     # e.g. for dis_steps=24, we have groups of 3
    #     keys = list(z.keys())[i:i+group_size]
    #     t = []
    #     for j in keys:
    #         # print("j:",j)
    #         t.append(z[j])
    #     running_total = t[0]
    #     local_cumsum = []
    #     local_cumsum.append(t[0].tolist())
    #     for chunk_val in range(1,len(t)):
    #         #get elementwise sum of element + next element 
    #         running_total = np.sum([running_total,(t[chunk_val][:len(running_total)])], axis=0)
    #         #append leftovers from next element
    #         tot = np.append(running_total, t[chunk_val][len(running_total):])
    #         running_total = np.append(running_total, t[chunk_val][len(running_total):])
    #         local_cumsum.append(tot)
    #     if(i==0):
    #         for b in local_cumsum:
    #             cumsum_z.append(b)
    #     else:
    #         cumsum_z.append(local_cumsum[-1])
    #cumsum_z is an array containing the cumulative sum of z through timesteps without values of z that shouldnt
    #be counted(their xval expired) it has shape of (6,zvals) 6 matches to each ts val in z dict
    #below does not do cumulative discards

    #below code gives an array of the cumsum with lifespan at each timestep for each server factored in
    cumsum_w_lifespan = np.reshape(np.array(x), (chunks, 28))
    dc1_cumsum = cumsum_w_lifespan[:, 0:7]
    dc2_cumsum = cumsum_w_lifespan[:, 7:14]
    dc3_cumsum = cumsum_w_lifespan[:, 14:21]
    dc4_cumsum = cumsum_w_lifespan[:, 21:28]
    dc_array = [dc1_cumsum,dc2_cumsum,dc3_cumsum,dc4_cumsum]
    ls_cumsum = []
    ls_cumsum2 = []
    for i in range(4):
        giga = []
        giga2= []
        step = 0
        for timestep in range(TIMESTEPS2):
            step=step%step_size
            modder = int(95/step_size)
            curr_chunk = int(timestep/step_size)
            modded_ts = curr_chunk%modder
            if(timestep >= 96):
                temp = dc_array[i][modded_ts:curr_chunk+1]#*(mult_array[step::step_size][::-1])
                temp2 = dc_array[i][modded_ts:curr_chunk+1]*(mult_array[step::step_size][::-1])
            else:
                mult_array[step-1:timestep+step_size-1:step_size]
                temp = dc_array[i][:curr_chunk+1]#*(mult_array[step:timestep+step_size:step_size][::-1])
                temp2 = dc_array[i][:curr_chunk+1]*(mult_array[step:timestep+step_size:step_size][::-1])
            temp = np.sum(temp, axis=0)
            temp2 = np.sum(temp2, axis=0)
            giga.append(temp)
            giga2.append(temp2)
            step+=1
        ls_cumsum.append(giga)
        ls_cumsum2.append(giga2)
    #array of datacenter and custom cumsum at each timestep shape(datacenter, timestep, servergen), 
    # expired servers are auto excluded
    ls_cumsum = np.array(ls_cumsum)
    # print(cumsum_z[2][0])
    # print("ls",ls_cumsum[0][100][0])
    ls_cumsum2 = np.array(ls_cumsum2)

    #get 1:1 match of buy to discard for the constraint of x-z=>0 later
    #NOTEE:just finding x-all_bound_to_x(Z)=>0 should be enough
    # all_timesteps_x_m_z = []
    # tt1 = x[:6*4*7] - cumsum_z[2]
    # tt2 = x[1*4*7:8*4*7] - cumsum_z[3]
    # tt3 = x[3*4*7:10*4*7] - cumsum_z[4]
    # tt4 = x[5*4*7:12*4*7] - cumsum_z[5]
    # all_timesteps_x_m_z.extend(tt1)
    # all_timesteps_x_m_z.extend(tt2)
    # all_timesteps_x_m_z.extend(tt3)
    # all_timesteps_x_m_z.extend(tt4)
    # print(len(all_timesteps_x_m_z))
    # print(all_timesteps_x_m_z[297])

    #cumulative discards might need to be done e.g. instead of x10-z10,x38-z66 do x10-z10,x10+x38-z10-z66
    reshaped_cumsum_z = np.reshape(cumsum_z,(12,4,7))
    for timestep in range(TIMESTEPS2):
        chunk = int(timestep/step_size)
        modder = int(96/step_size)
        if(timestep%step_size==0):
            for datacenter in range(4):
                if(timestep>=96 and timestep<144):
                    cumsum_no_expired = np.subtract(cumsum_x[chunk][datacenter], cumsum_x[chunk-modder][datacenter])
                    #get cumulative x -cumulative z
                    cumsum_x_m_z = cumsum_no_expired-reshaped_cumsum_z[chunk][datacenter]
                    solver.Add(np.sum(cumsum_x_m_z[:4]*2)+np.sum(cumsum_x_m_z[4:]*4)
                    <= dc_cap[datacenter])
                if(timestep<144):
                    cumsum_x_m_z = cumsum_x[chunk][datacenter]-reshaped_cumsum_z[chunk][datacenter]
                    #total used cap cannot be more than dc_cap
                    solver.Add(np.sum(cumsum_x_m_z[:4]*2)+np.sum(cumsum_x_m_z[4:]*4)
                    <= dc_cap[datacenter])
                if(timestep<=96):
                    solver.Add(np.sum(cumsum_x[chunk][datacenter][:4]*2)+np.sum(cumsum_x[chunk][datacenter][4:]*4)
                    <= dc_cap[datacenter])


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
                    #total used cap cannot be more than dc_cap
                    cumsum_cpu_no_expired = np.subtract(cumsum_x[chunk][datacenter][:4], cumsum_x[chunk-modder][datacenter][:4])
                    cumsum_gpu_no_expired = np.subtract(cumsum_x[chunk][datacenter][4:], cumsum_x[chunk-modder][datacenter][4:])
                    solver.Add(np.sum(cumsum_cpu_no_expired*2)+np.sum(cumsum_gpu_no_expired*4)
                    <= dc_cap[datacenter])

                if(datacenter != 3):
                    if(timestep<144):
                        for servergen in range(7):
                            index = timestep*21+datacenter*7+servergen
                            if(datacenter == 2):
                                x_minus_z_dc3 = ls_cumsum[datacenter][timestep][servergen]-reshaped_cumsum_z[chunk][datacenter][servergen]
                                x_minus_z_dc4 = ls_cumsum[datacenter+1][timestep][servergen]-reshaped_cumsum_z[chunk][datacenter+1][servergen]
                                solver.Add(x_minus_z_dc3*capacity[servergen]+x_minus_z_dc4*capacity[servergen] 
                                <= sens_demand[timestep][servergen])
                            else:
                                x_minus_z = ls_cumsum[datacenter][timestep][servergen]-reshaped_cumsum_z[chunk][datacenter][servergen]
                                #dc(xa-za)*cap<=demand
                                solver.Add(x_minus_z*capacity[servergen] <= sens_demand[timestep][servergen])
            else:
                solver.Add(np.sum(cumsum_x[chunk][datacenter][:4]*2)+np.sum(cumsum_x[chunk][datacenter][4:]*4)
                <= dc_cap[datacenter])
                for servergen in range(7):
                    index = timestep*21+datacenter*7+servergen
                    if(datacenter == 2):
                        solver.Add(ls_cumsum[datacenter][timestep][servergen]*capacity[servergen]
                            +ls_cumsum[datacenter+1][timestep][servergen]*capacity[servergen] <= sens_demand[timestep][servergen])
                    else:
                        solver.Add(ls_cumsum[datacenter][timestep][servergen]*capacity[servergen] 
                        <= sens_demand[timestep][servergen])

    #print("Number of constraints =", solver.NumConstraints())

    #solver.parameters.max_time_in_seconds = 10.0

    # Objective
    solver.Maximize(objective_func(demand2, x, z, ls_cumsum, step_size, TIMESTEPS2, START_STEP))
    
    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Total value =', make_readable(solver.Objective().Value()))
    else:
        print('The problem does not have an optimal solution.')

    if status == pywraplp.Solver.OPTIMAL:
        result_df = result_to_df(x,z,step_size,START_STEP2,TIMESTEPS2)
    else:
        return "no solution"
    
    return result_df

# %%
result_df = max_profit(pd.read_csv("../data/demand.csv"))
print(result_df)
result_df.to_csv('out.csv', index=True)
valid = verify_solution_integrity(result_df, True)
if(not valid):
    print("solution has an error!")
