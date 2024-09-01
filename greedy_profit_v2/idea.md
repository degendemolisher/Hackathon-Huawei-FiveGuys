# Greedy Profit Algorithm

Version 1 (too simple)
1) For each time step:
	1) Find the amount of demand of the current time step
	2) Calculate the number of servers to buy to satisfy that demand

Version 2 (this folder)
1) For each server/latency combination (in order of profitability):
	1) Find the ranges of time steps between which this server/latency is in demand
	2) Filter all ranges which last for less than the time it takes for the server/latency to break even
	3) For each range (from longest to shortest):
		1) Calculate the minimum demand across that range
		2) Calculate the number of servers to buy meet the minimum demand
		3) Store the number of servers to buy, which data centre, the buy time step, the dismiss time step
		4) For each demand in the range, subtract the capacity * number of servers to buy
	4) Repeat steps 1.1 to 1.3.4 with the new demand values until there are no ranges after 1.2
2) Convert the table representing the buy actions into actual buy actions

Things to account for
- Choose DC3 or DC4?
	- Default to put the most energy efficient 