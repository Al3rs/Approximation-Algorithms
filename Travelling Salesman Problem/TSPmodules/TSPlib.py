import timeit as t
import numpy as np
from mip import Model, minimize, BINARY, xsum 
from itertools import product

def TSPcostsMatrix(seed: int, n: int, maxCost: float):
    """""
    The function creates an undirected, simmetric and complete graph. It generates n points in [0, C_max/sqrt(2)] x [0, C_max/sqrt(2)] 
    and calculates the distances among them. 
    
    INPUT:
    seed = seed for a random number generator
    n = number of vertices
    maxCost = maximum distance beteween two vertices

    OUTPUT:
    costsMatrix = matrix of the distances/costs
    """

     # create the random number generator
    rng = np.random.default_rng(seed)

    # coordinates of the points generated
    x = np.zeros((n, 1))
    y = np.zeros((n, 1))

    # Matrix of the distances
    costsMatrix = float('nan') * np.ones((n, n))  

    # generate points in R^2
    x = np.array(rng.uniform(0, maxCost/np.sqrt(2), n)) 
    y = np.array(rng.uniform(0, maxCost/np.sqrt(2), n)) 

    # calculate the entries of the matrix
    for i in range(n):
        for j in range(n):
            if i < j:
                costsMatrix[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            else:
                costsMatrix[i, j] = costsMatrix[j, i]

    return costsMatrix



def PrimMST(C, rng):
    """"
    the function returns the minimum spanning tree obtained via Prim's algorithm.

    INPUT: 
    C = matrix of the distances
    rng = random number generator

    OUTPUT:
    MSTedges = list of the edges covered with the MST
    MSTcost = cost of the MST
    firstVertex = first vertex visited
    father = array containing father-son relationships among the vertices
    Visited =  list that represents the order in which vertices are visited
    time = computational time required to find a solution
    """
    n = np.size(C, 0) # number of vertices
    numVisited = 0 # number of vertices visited
    MSTcost = 0
    MSTedges = []
    Visited = []  

    key = float('inf') * np.ones(n, dtype = int)
    # if father[i] = j then node j is the father of node i
    father = list((-1) * np.ones((n, 1), dtype = int))

    start = t.default_timer()

    # select the first vertex to visit
    firstVertex = rng.integers(0, n, dtype = int)
    father[firstVertex] = -2
    key[firstVertex] = 0

    while numVisited < n:  # 4. repeat until all the vertices have been visited

        # 1.  Among the vertices that have not been visited, select the vertex that has the minimum key value
        minimumKey = float('inf')
        for i in range(n):
            if (i not in Visited) and (key[i] < minimumKey):
                minimumKey = key[i]
                minPosition = i

        # 2. add such vertex to set of the visited vertices
        Visited.append(minPosition)
        numVisited += 1

        # 3. update the value of the keys of the vertices that are adjacent to v and that have not been visited
        for v in range(n):
            if C[minPosition, v] < key[v] and (v not in Visited):
                key[v] = C[minPosition, v]
                father[v] = minPosition

        if numVisited == 1:
            u = minPosition
        elif numVisited == 2:  
            MSTedges.append((u, int(minPosition)))
            MSTcost += minimumKey
        else:
            MSTedges.append((int(father[minPosition]), minPosition))
            MSTcost += minimumKey

    stop = t.default_timer()

    return MSTedges, MSTcost, father, Visited, stop - start



def DepthFirstSearch(rng, v, father, Visited, NotVisited):
    """
    method that performs depth-first search in a given tree.
    
    INPUT:
    rng = random number generator
    v = a starting vertex
    father = array containing father-son relationships among the vertices
    Visited = list of the nodes already visited
    NotVisited = list of the nodes that have not been visited
    
    OUTPUT: None
    """
    n = len(father)
    rng.shuffle(NotVisited)
    cardVisited = len(Visited)
    
    if cardVisited < n:  
        for u in NotVisited:
            if father[u] == v:
                Visited.append(u)
                NotVisited.remove(u)
                cardVisited += 1
                DepthFirstSearch(rng, u, father, Visited, NotVisited)
                
            elif u == NotVisited[-1]:
                DepthFirstSearch(rng, father[v], father, Visited, NotVisited)

            

def HamiltonianCycle(rng, father, C, Visited):
    """
    Given a complete graph and a minimum spanning tree, the function utilizes the MST to construct a hamiltoninan cycle 
    using a depth-first-search algorithm.
    
    INPUT:
    rng = random number generator
    father = array that contains information about father-son relationships among the vertices of the graph
    C = matricìx of the costs/distances
    Visited = list that represents the order in which vertices were visited in the MST.
    
    OUTPUT:
    Tour = hamiltonian cycle
    costTour = cost of the tour
    time = computational time required to find a solution
    """
    
    costTour = 0
    NotVisited = Visited.copy() # list of the nodes that have not been visited
    Tour = [] # the cycle is represented as a list.
    
    start = t.default_timer()
    
    # add the root node of the tree as first node.
    Tour.append(Visited[0]) 
    NotVisited.remove(Visited[0])
    
    # create a path that explores all the nodes in the MST
    DepthFirstSearch(rng, Visited[0], father, Tour, NotVisited)
    
    # close the path to create a cycle
    Tour.append(Tour[0])
    
    # calculate the total cost
    for i in range(len(father)):
        costTour += C[Tour[i], Tour[i + 1]]
    
    stop = t.default_timer()
    
    return Tour, costTour, stop-start



def approx2_TSP(seed, C):
    """
    Given a complete graph with a matrix of the distances among nodes, the function outputs a feasable solution for the TSP.

    INPUT:
    seed = seed
    C = matrix of the distances/costs

    OUTPUT:
    Tour = hamiltonian cycle
    costTour = cost of the hamiltonian cycle
    time = computational time required to find a solution
    """
    # create the random number generator
    rng = np.random.default_rng(seed)    
    
    # find a MST
    MSTedges, MSTcost, father, Visited, runtimeMST =   PrimMST(C, rng)
    
    # find a hamiltonian cycle
    Tour, costTour, runtimeTour = HamiltonianCycle(rng, father, C, Visited)

    return Tour, costTour, runtimeTour, MSTedges, MSTcost, father, Visited, runtimeMST, runtimeMST+runtimeTour



class TravellingSalesmanProblem:
    
    def __init__(self, costsMatrix, seed = None):
        
        # check if the requirements are met
        assert costsMatrix.ndim == 2, "the matrix has more than two dimensions!"
        assert np.size(costsMatrix,0) == np.size(costsMatrix,1), "Dimensions do not coincide!"
        assert all(costsMatrix[i, j] >= 0 for i in range(np.size(costsMatrix, 0)) for j in range(np.size(costsMatrix, 1)) if i != j), "costs must be positive!"
        assert all(costsMatrix[i, j] < float('inf') for i in range(np.size(costsMatrix, 0)) for j in range(np.size(costsMatrix, 1)) if i != j), "The graph is not complete!"
        
        # Graph attributes
        self.costsMatrix = costsMatrix
        self.vertices = np.arange(np.size(costsMatrix,0))
        self.numVertices = np.size(costsMatrix,0)
        self.edges = [(i,j) for i in range(self.numVertices) for j in range(self.numVertices) if i!=j]        
        
        # attributes of the problem
        self.seed = seed
        
        
        
    def TSPsolve(self, mode: str):
        """
        the method creates a feasable solution for the TSP.
        
        INPUT:
        mode = solver that will be used to find a solution.(type "2 approximation " or "exact solver")
        """
        
        self.Tour = []
        self.costTour = 0
        self.totalRuntime = 0
        
        if mode == '2 approximation' and self.seed is not None:
            
            self.MSTedges = []
            self.MSTcost = 0
            self.Visited = []        
            self.father = [] 
            self.runtimeMST = 0
            self.runtimeTour =0
            
            # algorithm
            self.Tour, self.costTour, self.runtimeTour, self.MSTedges, self.MSTcost, self.father, self.Visited,self.runtimeMST, self.totalRuntime = approx2_TSP(self.seed, self.costsMatrix)
            
        elif mode == 'exact solver':
            
            V = set(self.vertices.copy())
            n = self.numVertices
            maxCost = np.nanmax(self.costsMatrix)
        
            for i in range(self.numVertices):
                self.costsMatrix[i][i] = maxCost**3 + 1
                
            model = Model()

            # binary variables indicating if arc (i,j) is used on the route or not
            x = [[model.add_var(var_type=BINARY) for j in range(n)] for i in range(n)]

            # continuous variable to prevent subtours: each city will have a
            # different sequential id in the planned route except the first one
            y = [model.add_var() for i in range(n)]

            # objective function: minimize the distance
            model.objective = minimize(xsum(self.costsMatrix[i][j]*x[i][j] for i in range(n) for j in range(n)))

            # constraint : leave each city only once
            for i in V:
                model += xsum(x[i][j] for j in V - {i}) == 1

            # constraint : enter each city only once
            for i in V:
                model += xsum(x[j][i] for j in V - {i}) == 1

            # subtour elimination
            for (i, j) in product(V - {0}, V - {0}):
                if i != j:
                    model += y[i] - (n+1)*x[i][j] >= y[j]-n

            # optimizing
            start = t.default_timer()
            model.optimize()
            stop = t.default_timer()
            
            OPTIM = 0
            OPTsolution = []
            for i in V:
                for j in V:
                    if x[i][j].x==1 :
                        OPTsolution.append((i,j))
                        OPTIM += self.costsMatrix[i][j]
                        
            self.Tour = OPTsolution
            self.costTour = OPTIM 
            self.totalRuntime = stop - start
            
        else:
            print('typed mode not recognized')
         
