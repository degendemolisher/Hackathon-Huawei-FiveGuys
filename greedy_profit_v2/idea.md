# Version 3 (current version)
- The same as version 2 but it handles the number of slots remaining in each datacenter

Core algorithm
1) Initialise a DataFrame that tracks the remaining slots of each datacentre at each time step
2) For each server/latency combination (in order of profitability):
	1) Find the ranges of time steps between which this server/latency is in demand
	2) Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)
	3) Filter all ranges which last for less than the time it takes for the server/latency to break even
	4) For each range (from longest to shortest):
		1) Calculate the minimum demand across that range
		2) Calculate the number of servers to buy meet the minimum demand
		3) Perform the Remaining Slot Decrement Algorithm
		4) For each demand in the range, subtract the capacity * number of servers to buy
		5) Filter new demand values which are too low to buy at least 1 server for
	5) Repeat steps 2.1 to 2.4.4 with the new demand values until there are no ranges after 2.3
3) Convert the table representing the results into correctly formatted buy and dismiss actions


Remaining Slot Decrement Algorithm (datacenter ID, server generation, remaining slots over a range of time steps, number of servers to buy)
1) Find the ranges of time steps between which the datacenter can fit at least 1 server
2) Filter ranges which last for less than the break even time
3) For each range (from longest to shortest):
    1) Calculate the minimum slot capacity
    2) Pick the minimum of the minimum slot capacity or the number of servers to buy * server slots size
    3) Store the number of servers to buy, the datacentre, the buy time step, the dismiss time step in the results
    4) Subtract the number of bought servers from the initial desired number of servers to buy
    5) For each slot capacity in the range for the datacenter, decrease the slot capacity by the result of step 3.2
5) Repeat steps 1 to 3.3 until there are no ranges left after step 2 or the all desired servers were purchased
	1) If there are no ranges and desired servers > 0 and the datacenter is DC3, change the datacenter to DC4 and repeat steps 1 to 3.3
6) Return a tuple of the (results, remaining_slots)


Suggestions:
- Step 2.4.4 assumes all the servers fit into the datacenters to ensure the loop actually finishes. Should this be adjusted?
- Still doesn't account for DC4


# Move strategy algorithm extension


After all the core algorithm is processed, you have 2 arrays representing the solution:
- The "results" of the algorithm (see remaining slot decrement algorithm step 3.3 for format)
- The "actions" of the algorithm (the actual format of a submitted solution)
1) For each server generation's results
	1) Sort results by operating time (dismiss time step - buy time step)
	2) Filter results with an operating time > life expectancy
	3) For each result (from shortest to longest server operating time)
		1) Search for a result of the same server generation which has a dismiss time step briefly before the this result's buy time step (based on some threshold e.g. >= 5 time steps before)
		2) Validate that the difference between the earlier buy time step and the later dismiss time step doesn't exceed the life expectancy (96)
		3) Calculate the number of servers to move: the minimum buy count of either result
		4) Find the server IDs of the servers to move from the actions array
		5) Construct move actions for each server to move to replace the buy/dismiss actions.
			1) If the datacenter moved from has better energy efficiency, perform the move at the same time step as the dismiss
			2) If the datacenter moved to has better energy efficiency, perform the move at the same time step as the buy
		6) Delete the corresponding buy and dismiss actions from the actions array



---

# OLD VERSIONS

# Version 1
1) For each time step:
	1) Find the amount of demand of the current time step
	2) Calculate the number of servers to buy to satisfy that demand

# Version 2
1) For each server/latency combination (in order of profitability):
	1) Find the ranges of time steps between which this server/latency is in demand
	2) Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)
	3) Filter all ranges which last for less than the time it takes for the server/latency to break even
	4) For each range (from longest to shortest):
		1) Calculate the minimum demand across that range
		2) Calculate the number of servers to buy meet the minimum demand
		3) Store the number of servers to buy, the datacentre, the buy time step, the dismiss time step
		4) For each demand in the range, subtract the capacity * number of servers to buy
		5) Filter new demand values which are too low to buy at least 1 server for
	5) Repeat steps 1.1 to 1.4.4 with the new demand values until there are no ranges after 1.3
2) Convert the table representing the results into correctly formatted buy and dismiss actions

Things to account for
- Choose DC3 or DC4?
	- Default to put the most energy efficient in DC4?
- What if there are 2 adjacent ranges with a small gap in between? All of the servers from the first range would be dismissed and re-purchased between ranges
	- See new step 1.2
- How do we handle when datacenters get full? How do we track when servers can be bought?
    - Once all results are processed, construct a data frame of the slots taken up by servers at each time step