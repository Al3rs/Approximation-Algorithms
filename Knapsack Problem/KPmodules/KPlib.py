import numpy as np
from mip import Model, BINARY, maximize, xsum
 
def KnapsackValuesWeights(s: int, n: int, w_min = 0.5, w_max = 10**6, v_min = 0.5, v_max = 10**6, C = 10**6):
    
    """
    The function generates items for a 0-1 knapsack problem.
    
    INPUT:
    n = number of items
    w_min = minimum weigth that an item can have
    w_max = maximum weight that an item can have
    p_min = minimum value that an item can have
    p_max = maximum value that an item can have
    s = seed
    C = capacity of the knapsack
    
    OUTPUT:
    v = array where the values of the items are stored
    w = array where the weights of the items are stored
    """
    
    # create the random number generator
    rng = np.random.default_rng(s)

    # generate weights and values of the items
    v = np.array(rng.uniform(v_min, v_max, n)) 
    w = np.array(rng.uniform(w_min, w_max, n)) 
    
    # make sure the istance created leads to a sensible problem
    if np.sum(w) <= C:
        k = rng.integers(0, n)
        w[k] = C

    return v, w

############################################ ALGORITHMS #########################################################################

# 2-APPROXIMATION ALGORITHM

import timeit as t

def approx2_KP(v, w, C):
    
    """
    given a set of items with values v and weights w, the function returns a feasable solution of the knapsack problem where the knapsack has 
    capacity C. The approximation ratio is 2.

    INPUT:
    v = array where the values of the items are stored
    w = array where the weights of the items are stored
    C = capacity of the knapsack

    OUTPUT:
    Solution = set containing the item selected
    weightS = total weigth of the item selected
    valueS =  total value of the item selected
    time = computational time required to find a solution 
    """

    # INITIALIZATIONS
    Solution = set() # set of the item selected
    weightS = 0 # weight of the solution
    valueS = 0 # value of the solution
    

    # item with the highest value
    itemMaxValue = np.argmax(v)
    MaxValue = np.max(v)

    # calculate the density
    d = v/w

    start = t.default_timer()

    # sort the items based on the density 
    indices = np.argsort(d)[::-1]

    i = 0
    while weightS + w[indices[i]] <= C:
        weightS += w[indices[i]]
        valueS += v[indices[i]]
        Solution.add(indices[i])
        i += 1

    if valueS < MaxValue and w[itemMaxValue] <= C :
        Solution.clear()
        Solution.add(itemMaxValue)
        weightS = w[itemMaxValue]
        valueS = MaxValue

    stop = t.default_timer()

    return Solution, weightS, valueS, stop - start

################################################################################################################################

# PTAS ALGORITHM

def PTAS_KP(epsilon, v, w, C):
    """"
    given a set of items with values v and weights w, the function returns a feasable solution of the knapsack problem where 
    the knapsack has capacity C. The approximation ratio is 1+epsilon.
 
    INPUT:
    epsilon = maximum percentage error with respect to the optimal solution
    v = array where the values of the items are stored
    w = array where the weights of the items are stored
    C = capacity of the knapsack

    OUTPUT:
    Solution = set containing the item selected
    weightS = total weigth of the item selected
    valueS =  total value of the item selected
    time = computational time required to find a solution
    """

    n = len(v) # number of items
    Solution = set() 
    weightS = 0
    valueS = 0   
    K = epsilon * np.max(v) / n # scale factor
    roundedV = np.zeros((n, 1), dtype = int)  # rounded values
    totalRoundedValues = 0  # sum of all rounded values
    OPT = 0 # OPT+1 is the optimal value after rescaling and rounding the values of the items 

    start1 = t.default_timer()

    # round and scale the values of the items
    roundedV = np.floor(v / K).astype(int)
        
    stop1 = t.default_timer()

    #roundedV = np.array(roundedV, dtype=int)
    totalRoundedValues = np.sum(roundedV)
    roundedV_min = np.min(roundedV)

    # Given the first i+1 items, A[i,j] = minimum weight needed to obtain a knapsack with total value = j
    # A = C**2 if such knapsack does not exist 
    A = (C**2)*np.ones((n+1,  totalRoundedValues + 1)) 

    for i in range(n + 1):
        A[i, 0] = 0  

    # binary decision matrix: given the first i items, X[i,j] = 1 if the i-th item is included in the minimum-weight knapsack that has 
    # a total value equal to j.
    X = np.zeros((n+1,  totalRoundedValues+ 1))

    start = t.default_timer()

    for i in np.arange(1, n + 1, dtype=int):  

        for j in np.arange(roundedV_min, totalRoundedValues + 1):
            # compare the weight of two sets that are composed of at most i items and have a total value of j:
            # 1) the set obtained by adding the i-th item to the set composed of the first i-1 items that 
            #    has a value of j-roundedV[i-1];
            # 2) the set composed of the first i-1;
            # choose the set with the minimum weight: A[i,j] = min(w[i - 1] + A[i - 1, j - roundedV[i - 1]],A[i - 1, j])  
            if (j - roundedV[i - 1] >= 0) and (w[i - 1] + A[i - 1, j - roundedV[i - 1]] < A[i - 1, j]):  
                A[i,j] = w[i - 1] + A[i - 1, j - roundedV[i - 1]]
                X[i,j] = 1
            else:
                # in this case the minimum weight needed to have a total value equal to j was reached using i-1 items 
                A[i, j] = A[i - 1, j]  
                    
            if A[i, j] <= C and j > OPT:
                OPT = j

    for i in np.arange(n+1)[::-1]:
        if X[i, OPT] == 1:
            Solution.add(i-1)
            valueS += v[i-1]
            weightS += w[i-1]
            OPT = OPT - roundedV[i-1]

    stop = t.default_timer()

    return Solution, weightS, valueS, stop + stop1 - start - start1



class KnapsackProblem:
    
    def __init__(self, v, w, C, epsilon = None):
        # check if the requirements are met
        assert np.all(v >= 0), "values must be positive!"
        assert np.all(w > 0), "weigths must be STRICTLY positive!"
        assert np.min(w)< C, "no item can be put in the knapsack!"
        if epsilon != None:
            assert epsilon > 0 and epsilon < 1, "epsilon must be STRICTLY greater than zero AND STRICTLY less than one!"
        
        # attributes
        self.values =  v
        self.weights = w
        self.capacity = C
        self.epsilon = epsilon
        self.itemList = [(i, w[i], v[i]) for i in range(len(self.values))]
        
    def KPSolver(self, mode: str):
        """
        the method creates a feasable solution for the KP.
        
        INPUT:
        mode = solver that will be used to find a solution.(type "2 approximation ", "PTAS" or "exact solver")
        """
        self.Solution = []
        self.weightS = 0
        self.valueS = 0
        self.runtime = 0
        
        if mode == "2 approximation":
            self.Solution, self.weightS, self.valueS, self.runtime = approx2_KP(self.values, self.weights, self.capacity)  
            
            
        elif mode == "PTAS":
            self.Solution, self.weightS, self.valueS, self.runtime = PTAS_KP(self.epsilon, self.values, self.weights, self.capacity)
            
            
        elif mode == "exact solver":
            I = range(len(self.values))

            m = Model("knapsack")

            x = [m.add_var(var_type=BINARY) for i in I]

            m.objective = maximize(xsum(self.values[i] * x[i] for i in I))

            m += xsum(self.weights[i] * x[i] for i in I) <= self.capacity
            
            start = t.default_timer()
            m.optimize()
            stop = t.default_timer()
            
            self.runtime = stop - start
            
            self.Solution = {i for i in I if x[i].x >= 0.99}
            
            for i in self.Solution:
                self.valueS += self.values[i]
                self.weightS += self.weights[i]
            
            
        else:
            print("typed mode not recognized")
            
        
        
        