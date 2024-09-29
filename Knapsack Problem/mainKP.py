from mip import Model, xsum, maximize, BINARY
from KPmodules.KPlib import *

# INPUT PARAMETERS
n = 10 # number of items
seed = 0 
C = n ** 2 # capacity of the knapsack
w_min = 1 # minimum weight
w_max = C # maximum weight
v_min = 0 # minimum value
v_max = n ** 3 / 2 # maximum value

# GENERATE VALUES AD WIEGHTS
v, w = KnapsackValuesWeights(seed, n, w_min, w_max, v_min, v_max, C)
print('\nItems : ')
print('weigths = ',w)
print('values = ',v)
print('knapsack capacity = ',C)

#######################################################

# 2-APPROXIMATION
# generate an instance
KP2approx = KnapsackProblem(v, w, C)
# solve 
KP2approx.KPSolver('2 approximation')
# OUTPUT 2-approx
print("\n\nRESULTS OF THE 2-APPROXIMATION ALGORITHM:")
print(f"Item selected = {KP2approx.Solution}")
print(f"Value of the knapsack = {KP2approx.valueS}")
print(f"Weigth of the knapsack = {KP2approx.weightS}")
print(f"Runtime = {KP2approx.runtime}")

#####################################################################

#FPTAS
epsilon = 0.01
# generate an instance
KPptas = KnapsackProblem(v, w, C, epsilon)
# solve 
KPptas.KPSolver('PTAS')
# OUTPUT 2-approx
print(f"\n\nRESULTS OF THE PTAS WITH EPSIILON = {epsilon}:")
print(f"Item selected = {KPptas.Solution}")
print(f"Value of the knapsack = {KPptas.valueS}")
print(f"Weigth of the knapsack = {KPptas.weightS}")
print(f"Runtime = {KPptas.runtime}")

####################################################################

# EXACT SOLVER
# generate an instance
KPexact = KnapsackProblem(v, w, C)
# solve 
KPexact.KPSolver('exact solver')
# OUTPUT 2-approx
print("\n\nRESULTS OF THE EXACT SOLVER:")
print(f"Item selected = {KPexact.Solution}")
print(f"Value of the knapsack = {KPexact.valueS}")
print(f"Weigth of the knapsack = {KPexact.weightS}")
print(f"Runtime = {KPexact.runtime}")

####################################################################

# COMPARISON
print("\n\n COMPARISONS:")
print(f"Percentage error of the 2-approximation = {1- KP2approx.valueS / KPexact.valueS}")
print(f"Percentage error of the PTAS with epsilon equal to {epsilon} = {1- KPptas.valueS / KPexact.valueS}")