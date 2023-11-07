#suppose both node1 and node2 are monitoring node0
#we have the traces node1 receive from node0, and traces node2 receive from node0, node1 and node2 would form a list storing when they suspect node0
#we have traces sending from node1 to node2, according to these traces, node1 'transports' its information about node0 to node2

trace0_1 = [1.0,2.2,3.0,4.0,5.7,7.0,8.0] #for example, these are the traces node1 received from node0
suspect1 = [(0.0,1.0),(2.0,2.2),(5.0,5.7),(6.7,7.0)] #these are the time intervals that node1 suspects node0

trace0_2 = [1.0,2.5,3.5,4.7,6.0,6.7,7.5]
suspect2 = [(0.0,1.0),(2.0,2.5),(4.5,4.7),(5.5,6.0)]

trace1_2 = [(0.0,1.0),(1.0,2.2),(2.0,2.6),(3.0,4.1),(4.0,5.0),(5.0,6.1),(6.0,7.2)] #these are traces sent fron node1 to node2

info1_2 = [[(0.0,0.0),'s'],[(0.0,1.0),'s'],[(1.0,2.0),'t'],[(2.0,3.0),'t'],[(3.0,4.0),'t'],[(4.0,5.0),'t'],[(5.0,6.0),'t']] #information about node0 node2 received from node1
#info1_2 can be easily generated using computer programs

#combining suspect2 and info1_2, we can have the final suspection result in at node2, and we can calculate new metric results based on this suspection info