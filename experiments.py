import pandas as pd
import numpy as np
import torch
from trop_fns import *
import random
from time import time
from WDDD_class import *

def mean_print(exps_DF, settings, names):
    
    exps_DF
    
    for alpha in names[0]:
        for exp in names[1]:
            exp_DF = exps_DF[(exps_DF[settings[0]] == alpha) & (exps_DF[settings[1]] == exp)]
            print(f"Time taken for {exp}, {alpha}: {np.mean(exp_DF['time']):.2f} seconds. Mean, STD for {exp}, {alpha}: {np.mean(exp_DF['end_loss']):.2f} plus/minus {np.std(exp_DF['end_loss']):.2f}")
            
def log_error(DF):
                  
    solution_DF = DF[['loss_values', 'N', 'M', 'N_data_type', 'M_data_type']].copy()
    solution_DF['optimal_val'] = DF.apply(lambda row: min(row['loss_values']), axis=1)
    solution_DF = solution_DF.groupby(['N', 'M', 'N_data_type', 'M_data_type'])['optimal_val'].agg([('min', 'min')]).reset_index()
    
    log_error_DF = DF.merge(solution_DF, on = ['N', 'M', 'N_data_type', 'M_data_type'])
                  
    DF['log_error']  = log_error_DF[['loss_values', 'min']].apply(lambda x: error(x['loss_values'], x['min']), axis=1)
    DF['end_log_error'] = DF['log_error'].str[-1]
                  
    return DF
                  
def error(loss_values,global_soln = 0.8, relative = True):
    if relative:
        return np.log(np.array(loss_values) - global_soln*0.99) - np.log(global_soln*0.99)
    else:
        return np.log(np.array(loss_values) - global_soln + 0.00001)
                      
def make_DF(R_list = [50], 
                        NM_list = [[10,6],[15,6], [15,10]], 
                        type_list = ['branching', 'coalescent', 'gaussian'], 
                        grad_list = ["CD", "TD", "CS", "TS", "CA", "TA"], 
                        num_exps = 20, 
                        lr_df = None,
                        scale_df = None,
                        graph_list = ["comp", "incomp"],
                        p = ["infty"],
                        steps = [20],
                        shift_num_steps = [20],
                        supp_num_steps = [20],
                        normalised_data = True
                       ):
    
    #Form data frame columns with sample data details - sample size(data_count), dimensionality(data_dim), sample measure type(data_type), and an empty column for sample data(data)
    R_df = pd.DataFrame(R_list, columns = ['R'])
    NM_df = pd.DataFrame(NM_list, columns = ['N', 'M'])
    N_type_df = pd.DataFrame(type_list, columns = ['N_data_type'])
    M_type_df = pd.DataFrame(type_list, columns = ['M_data_type'])
    empty_data_df = pd.DataFrame([[None, None]], columns = ['high_data', 'low_data'])
    p_df = pd.DataFrame(p, columns = ['p'])
        
    data_df = R_df.merge(NM_df, how = "cross").merge(N_type_df, how = "cross").merge(M_type_df, how = "cross").merge(empty_data_df, how = "cross").merge(p_df, how = "cross")
    
    #Filling in the data column with sampled data
    for index, row in data_df.iterrows():
        if row['N_data_type'] == 'branching':
            X = np.load(f"Data/Norm{row['N']}.0_R50.0.npy")[:row['R'],:]
        elif row['N_data_type'] == 'coalescent':
            X = np.load(f"Data/Coal{row['N']}.0_R50.0.npy")[:row['R'],:]
        elif row['N_data_type'] == 'gaussian': 
            X = np.load(f"Data/Gaus{row['N']}.0_R50.0.npy")[:row['R'],:]
        if row['M_data_type'] == 'branching':
            Y = np.load(f"Data/Norm{row['M']}.0_R50.0.npy")[:row['R'],:]
        elif row['M_data_type'] == 'coalescent':
            Y = np.load(f"Data/Coal{row['M']}.0_R50.0.npy")[:row['R'],:]
        elif row['M_data_type'] == 'gaussian': 
            Y = np.load(f"Data/Gaus{row['M']}.0_R50.0.npy")[:row['R'],:]
        
        if normalised_data:
            X = torch.tensor(X/np.mean(np.max(X,1) - np.min(X,1)))
            Y = torch.tensor(Y/np.mean(np.max(Y,1) - np.min(Y,1)))
        else:
            X = torch.tensor(X)
            Y = torch.tensor(Y)
        data_df.at[index, 'high_data'] = X
        data_df.at[index, 'low_data'] = Y

    exp_df = pd.DataFrame(range(num_exps), columns = ['num_exp'])
    empty_init_df = pd.DataFrame([[None, None]], columns = ['t_init', 'supp_init'])                
    init_df = exp_df.merge(empty_init_df, how = "cross").merge(NM_df, how = "cross")
                           
    for index, row in init_df.iterrows():
        init_df.at[index, 't_init'] = torch.randn(row['N'])
        init_df.at[index, 'supp_init'] = rand_supp(row['N'], row['M'])
                       
    DF = pd.merge(init_df, data_df, on = ['N', 'M'])
                        
    shift_num_steps_df = pd.DataFrame(shift_num_steps, columns = ['shift_num_steps'])
    supp_num_steps_df = pd.DataFrame(supp_num_steps, columns = ['supp_num_steps'])
    steps_df = pd.DataFrame(steps, columns = ['steps'])
    grad_df = pd.DataFrame(grad_list, columns = ['grad'])
    graph_df = pd.DataFrame(graph_list, columns = ['graph'])
    results_df = pd.DataFrame([[None, None]], columns = ['loss_values', 'time_taken'])
                
    DF = DF.merge(grad_df, how = "cross").merge(graph_df, how = "cross").merge(results_df, how = "cross").merge(steps_df, how = "cross").merge(supp_num_steps_df, how = "cross").merge(shift_num_steps_df, how = "cross")
    
    if lr_df is not None:
        DF = DF.merge(lr_df, on = ['data_dim', 'data_count', 'grad'])
                        
    if scale_df is not None:
        DF = DF.merge(scale_df, on = ['data_dim', 'data_count', 'graph'])
    
    return DF

def experiment(DF, verbose_bool = True):

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)                    
                        
    for index, row in DF.iterrows():
                        
        Dist = WDDD(row['high_data'],row['low_data'], p=row['p'])
                        
        init_shift = row['t_init'].clone().detach().requires_grad_(True)
        init_probs = torch.full((row['R'],row['R']),1/(row['R']*row['R']))
                        
        Dist.initialise(init_shift, row['supp_init'], init_probs)

        start = time()

        loss = Dist.minimise(row['supp_num_steps'], row['shift_num_steps'], row['steps'], lr = row['lr'], scale = row['scale'], grad = row['grad'], graph = row['graph'], verbose = verbose_bool)

        stop = time()

        DF.at[index, 'time_taken'] = stop - start
        DF.at[index, 'loss_values'] = torch.tensor(loss).detach().numpy()
    
    return DF