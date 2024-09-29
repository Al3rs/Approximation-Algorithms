from TSPmodules.TSPlib import *

# INPUT PARAMETERS
n = 8  # number of vertices
C_max = n**3/2
seed = 0

# create the matrix of the costs/distances
C = TSPcostsMatrix(seed, n, C_max)
print(f"Matrix of the distances: \n{C}")

#############################################################################################
# 2 - APPROXIMATION

# create an instance of the problem
tsp2approx = TravellingSalesmanProblem(C, seed)
# solve the problem
tsp2approx.TSPsolve('2 approximation')
# print the results
print("\n\n2-APPROXIMATION RESULTS:")
print(f"\nPath = {tsp2approx.Tour}")
print(f"Total cost = {tsp2approx.costTour}")
print(f"Runtime = {tsp2approx.totalRuntime}")
print("\n\n")


#############################################################################################
# EXACT SOLUTION 
# create an instance of the problem
tspExact = TravellingSalesmanProblem(C, seed)
# solve the problem
tspExact.TSPsolve('exact solver')
# print the results
print("EXACT SOLVER RESULTS:")
print(f"\nPath = {tspExact.Tour}")
print(f"Total cost = {tspExact.costTour}")
print(f"Runtime = {tspExact.totalRuntime}")
print("\n\n")
            
#############################################################################################
# COMPARISON
print(f"Percentage error = {tsp2approx.costTour/tspExact.costTour - 1}")