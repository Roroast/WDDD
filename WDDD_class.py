import torch
import copy
from trop_fns import *
import math
import torch.optim as optim
import ot
import random
from time import time

class WDDD:

    def __init__(self, Xdataset, Ydataset, p=2):
        if Xdataset.shape[1] < Ydataset.shape[1]:
            self.S, self.N = Ydataset.shape
            self.R, self.M = Xdataset.shape
            self.low_dataset = Xdataset.clone().detach()
            self.high_dataset = Ydataset.clone().detach()
        else:
            self.S, self.N = Xdataset.shape
            self.R, self.M = Ydataset.shape
            self.low_dataset = Ydataset.clone().detach()
            self.high_dataset = Xdataset.clone().detach()
        self.p = p

        self.high_weights = torch.ones(self.S, requires_grad = False) / torch.tensor(self.S, requires_grad = False)
        self.low_weights = torch.ones(self.R, requires_grad = False) / torch.tensor(self.R, requires_grad = False)

        self.shift = None
        self.support = None
        self.probs = None
        self.loss = None
        self.C = None
    
    def initialise(OP, shift = None, support = None, probs = None):
        if torch.is_tensor(shift):
            OP.shift = shift.clone().detach().requires_grad_(True)
        elif (OP.shift is None):
            OP.shift = torch.zeros(OP.N, requires_grad = True)

        if isinstance(support, list):
            OP.support = copy.copy(support)
        elif OP.support is None:
            OP.support = rand_supp(OP.N,OP.M)

        if torch.is_tensor(probs):
            OP.probs = probs.clone().detach()
        elif probs is not None:
            OP.probs = torch.tensor(probs)
        elif OP.probs is None:
            OP.probs = torch.full((OP.R,OP.S),1/(OP.R*OP.S))
        
        _,OP.loss = objective_function(OP.high_dataset,OP.low_dataset,OP.support,OP.shift,OP.probs,OP.p) 

    def train_shift(OP, t, supp = None, lr = 1, num_steps = 50, tol_loss = 0.1, tol_mat = 0.05, termination = True, grad = 'CA', verbose = False):
    #Implements gradient descent to train shift.
    #Params:    alpha: Initial step size. Step size decreases harmonically over susequent iterations.
    #           max_iters: Maximum number of iterations.
    #           tol_loss: convergence criterion for reaching zero
    #           tol_mat: convergence criterion for no changes in shift
    #           prob_updates: rate of updating OP.probs
    #           termination: allowing termination by convergence or not
    #           grad_type: the step direction type
    #           verbose: returning the location and loss of every step or not
    
        N = len(t.data)
        if grad in ["CS", "TS"]:
            num_steps = math.floor(1.9*num_steps)
            
        if supp is None:
            supp = OP.support
        
        steps = num_steps

        loss_values = []
        t_values = []

        if grad in ["CA"]:
            optimizer = optim.Adamax([t], lr=lr)
        elif grad in ["TA"]:
            optimizer = optim.Adamax([t], lr=lr)
        elif grad in ["CS", "TS", "CB", "TB"]:
            optimizer = optim.SGD([t], lr = lr)
            
        # Perform gradient descent
        for i in range(num_steps):
            
            _,loss = objective_function(OP.high_dataset,OP.low_dataset,supp,t,OP.probs,OP.p)
            loss_values.append(loss.item())
            t_values.append(t.detach().clone().numpy())
            loss.backward()
            
            #step direction computation
            if not torch.nonzero(t.grad.data).numel():
                if grad in ["CS", "TS"]:
                    continue
                else:
                    steps = i
                    OP.shift = t
                    break
            elif grad in ["CD", "TD", "CS", "TS"]:
                if grad in ["CD", "CS"]:
                    step = (t.grad.data)/torch.linalg.norm(t.grad.data, ord = 2)
                    norm = torch.linalg.norm(t.grad.data, ord = 2)
                else:
                    step = (t.grad.data > 0).float()
                    norm = torch.max(t.grad.data) - torch.min(t.grad.data)
                t.data = t.data - lr * step * norm / torch.sqrt(torch.tensor(i+1))
            elif grad in ["CA"]:
                optimizer.step()
            elif grad in ["TA", "TS"]:
                t.grad.data = (t.grad.data > 0).float() * (torch.max(t.grad.data) - torch.min(t.grad.data))
                optimizer.step()

            t.grad.data.zero_()

            #convergence tests
            if i>5 and termination:
                if (np.abs(loss_values[-1] - loss_values[-4]) < tol_loss):
                    steps = i
                    OP.shift = t
                    break
                    
        OP.C, OP.loss = objective_function(OP.high_dataset, OP.low_dataset, OP.support, OP.shift, OP.probs, OP.p)
              
        steps = i
        OP.shift = t
        
        if verbose:
            return loss_values, t
        else:
            return loss_values[-1], t

    def train_probs(OP):
        OP.probs = torch.tensor(ot.emd(np.array(OP.high_weights), np.array(OP.low_weights), OP.C.detach().numpy()), requires_grad = False)
        OP.C, OP.loss = objective_function(OP.high_dataset, OP.low_dataset, OP.support, OP.shift, OP.probs, OP.p)
        return OP.loss
        
    def get_neigh(OP, supp): #get random neighbour
        no_neighbour = True
        while no_neighbour:
            u = np.random.rand(1)
            Neighs = NoNeighs(supp)
            threshold = Neighs/(math.factorial(OP.M)+ Neighs)
            if u < threshold:
                #change col
                n = random.choice(range(OP.N))
                prev_pos = np.argmax([n in supp[i] for i in range(len(supp))])
                if len(supp[prev_pos]) == 1:
                    continue
                m = random.choice([i for i in range(OP.M) if i != prev_pos])
                neigh_supp = copy.deepcopy(supp)
                neigh_supp[prev_pos].remove(n)
                neigh_supp[m].add(n)
                no_neighbour = False
            else:
                #permute rows
                [a,b] = random.sample(range(OP.M),2)
                neigh_supp = copy.deepcopy(supp)
                neigh_supp[a], neigh_supp[b] = neigh_supp[b], neigh_supp[a]
                no_neighbour = False

        return neigh_supp
            
    def train_supp(OP, graph = "comp", grad = "TA", num_steps = 100, shift_num_steps = 10, scale = 1, lr = 0.1, error = 0.1, verbose = False):

        supp = copy.deepcopy(OP.support)
        loss = OP.loss.detach()

        if verbose:
            tracking = []

        iter = 0

        while iter < num_steps:

            T = 1 - iter/num_steps

            if graph == "comp":
                neigh_supp = rand_supp(OP.N, OP.M)
            else:
                neigh_supp = OP.get_neigh(supp)
                
            t = OP.shift.detach().requires_grad_()
            
            _, neigh_loss = objective_function(OP.high_dataset, OP.low_dataset, neigh_supp, OP.shift, OP.probs, OP.p)
                
            move = False
            u = random.random()
            if np.exp((loss-neigh_loss).detach().numpy()/(T*scale)) > u:
                move = True
            
            if move:
                supp = copy.deepcopy(neigh_supp)
                loss = neigh_loss
            
            iter += 1

            if verbose:
                tracking+= [loss]

        OP.support = supp

        if verbose:
            OP.C, OP.loss = objective_function(OP.high_dataset, OP.low_dataset, OP.support, OP.shift, OP.probs, OP.p)
            return tracking
        else:
            OP.C, OP.loss = objective_function(OP.high_dataset, OP.low_dataset, OP.support, OP.shift, OP.probs, OP.p)
            return OP.loss

    def minimise(OP, supp_num_steps, shift_num_steps, steps, carried_shift = True, lr = 0.1, scale = 1, grad = 'T1', graph = "comp", verbose = False):

        #OP.initialise()
        
        if verbose:
            track = []
            for step in range(steps):
                if carried_shift:
                    init_t = OP.shift
                else:
                    init_t = torch.rand(OP.N, requires_grad = True)
                shift_run = OP.train_shift(init_t,  lr = lr/np.sqrt(step+1), num_steps = shift_num_steps, termination = False, grad = grad, verbose = True)
                track = track + shift_run[0]
                if supp_num_steps > 0:
                    track.append(OP.train_probs())
                    supp_run = OP.train_supp(num_steps = supp_num_steps, scale = scale/np.sqrt(step+1), graph = graph, verbose = True)
                    track = track + supp_run
        else:
            for step in range(steps):
                OP.train_shift(OP.shift, lr = lr/np.sqrt(step+1), num_steps = shift_num_steps, grad = grad)
                OP.train_probs()
                OP.train_supp(num_steps = supp_num_steps, scale = scale/np.sqrt(step+1), graph = graph)
                
        if verbose:
            return track