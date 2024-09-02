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


# Version 3 (CURRENT VERSION)
- The same as version 2 but it handles the number of slots remaining in each datacenter

1) Initialise a DataFrame that tracks the remaining slots of each datacentre at each time step
2) For each server/latency combination (in order of profitability):
	1) Find the ranges of time steps between which this server/latency is in demand
	2) Merge ranges which have a negligibly small gap in between (relative to the length of the smallest range)
	3) Filter all ranges which last for less than the time it takes for the server/latency to break even
	4) For each range (from longest to shortest):
		1) Calculate the minimum demand across that range
		2) Calculate the number of servers to buy meet the minimum demand
		3) For the time steps in this range, decrease the remaining slots in the appropriate datacenter by the number of servers * slots size
			1) If all the servers can't fit in the datacenter for the entire range:
				1) Find the minimum number of remaining slots for the range and only buy servers for those slots.
		4) Store the number of servers to buy, the datacentre, the buy time step, the dismiss time step in the results
		5) For each demand in the range, subtract the capacity * number of servers to buy
		6) Filter new demand values which are too low to buy at least 1 server for
	5) Repeat steps 1.1 to 1.4.4 with the new demand values until there are no ranges after 1.3
3) Convert the table representing the results into correctly formatted buy and dismiss actions


Suggestions:
- For a move strategy (to increase server lifespan)
	- After all the results are processed, scan for buy actions which occur within a certain number of time_steps from a dismiss action for the same type of server.
		- If the datacenter moved from has better energy efficiency, replace the buy action with a move action and delete the dismiss action
		- If the datacenter moved to has better energy efficiency, replace the dismiss action with a move action and delete the dismiss action
- Improvements for 2.4.3.1?



Example remaining slots DataFrame:

| TS  | DC1   | DC2   | DC3  | DC4  |
| --- | ----- | ----- | ---- | ---- |
| 1   | 21998 | 13000 | 7000 | 8000 |
| 2   | 21998 | 13000 | 7000 | 8000 |
| 3   | 22000 | 13000 | 7000 | 8000 |
| ... |       |       |      |      |
| 168 | 22000 | 13000 | 7000 | 8000 |
