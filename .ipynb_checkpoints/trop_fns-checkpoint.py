import numpy as np
import torch
import itertools
import copy

# random support from code

def rand_supp(N, M):
    #Generating a random map
    permutation = np.random.permutation(range(N)).tolist()
    coords = list(range(M)) + list(np.random.choice(range(M), N-M))

    supp = [set() for n in range(M)]

    for n in range(N):
        supp[coords[n]].add(permutation[n])
    
    return supp

# Define the objective function
def objective_function(X, Y, supp, t, probs, p, subsample = None):
    # X is input data of dim N
    # Y is output data of dim M

    obj_func = 0
    C = torch.zeros((len(X),len(Y)))

    if subsample is not None:
        X_sample = subsample[0]
        Y_sample = subsample[1]
    else:
        X_sample = range(len(X))
        Y_sample = range(len(Y))
    obj_func = 0
    C = torch.zeros((len(X),len(Y)))
    b = torch.zeros((len(X),len(supp)))
    
    for j, row_supp in enumerate(supp):
        rows = t[list(row_supp)]+X[:,list(row_supp)]
        b[:,j] = torch.max(rows, dim = 1).values
    Z = torch.unsqueeze(Y, 1)-torch.unsqueeze(b, 0)
    C = (torch.max(Z, dim = 2).values - torch.min(Z, dim = 2).values)
        
    if isinstance(p, int):
        return C, torch.sum(torch.mul(C**p,probs))**(1/p)
    elif p == "infty":
        return C, torch.max(C*torch.sign(probs))

# Defining projections for visualisations
def proj(X):
    proj_mat = torch.tensor([[1, -0.5, -0.5],[0, np.sqrt(3)/2, -np.sqrt(3)/2]],dtype = torch.double)
    return torch.matmul(proj_mat, X)/torch.sqrt(torch.tensor(1.5))

def rev_proj(X):
    rev_proj_mat = torch.tensor([[1, 0],[-1/2, np.sqrt(3)/2],[-1/2, -np.sqrt(3)/2]],dtype = torch.double)
    return torch.t(torch.matmul(rev_proj_mat, X)/torch.sqrt(torch.tensor(1.5)))

def NoNeighs(supp):
    return np.sum(np.array([len(J) -1 for J in supp]) * (len(supp) - 1))