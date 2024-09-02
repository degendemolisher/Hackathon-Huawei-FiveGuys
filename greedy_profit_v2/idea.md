# Greedy Profit Algorithm

Version 1 (too simple)
1) For each time step:
	1) Find the amount of demand of the current time step
	2) Calculate the number of servers to buy to satisfy that demand

Version 2 (this folder)
1) For each server/latency combination (in order of profitability):
	1) Find the ranges of time steps between which this server/latency is in demand
	2) Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)
	3) Filter all ranges which last for less than the time it takes for the server/latency to break even
	4) For each range (from longest to shortest):
		1) Calculate the minimum demand across that range
		2) Calculate the number of servers to buy meet the minimum demand
		3) Store the number of servers to buy, which data centre, the buy time step, the dismiss time step
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