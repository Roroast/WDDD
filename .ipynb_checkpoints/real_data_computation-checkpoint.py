import pandas as pd
from trop_fns import *
from WDDD_class import *
from experiments import *

real_datasets = []
for n in range(1993, 2014):
    X = np.load(f"Data/N_NYh3n2_HA_20000_5/{n}.npy")[:50,:]
    real_datasets.append(X)
    
datafile_df = pd.DataFrame([[str(n)+".npy", str(m)+".npy", None, None] for m in range(1993, 2014) for n in range(1993, m+1)], columns = ["high_datafile", "low_datafile", "high_data", "low_data"])

for index, row in datafile_df.iterrows():
        X = np.load("Data/N_NYh3n2_HA_20000_5/"+row["high_datafile"])[:50,:]
        Y = np.load("Data/N_NYh3n2_HA_20000_5/"+row["low_datafile"])[:50,:]
        
        row["high_data"] = torch.tensor(X/np.mean(np.max(X,1) - np.min(X,1)))
        row["low_data"] = torch.tensor(Y/np.mean(np.max(Y,1) - np.min(Y,1)))

hyperparam_df = pd.DataFrame([["TA", "incomp", 2, 40, 50, 50, np.e**(-1), np.e**(-2), 50, None, None]], columns = ["grad", "graph", "p", "steps", "supp_num_steps", "shift_num_steps", "lr", "scale", "R", "time_taken", "loss_values"])

num_exps = 10

empty_init_df = pd.DataFrame([[None, None]], columns = ['t_init', 'supp_init'])  
exp_df = pd.DataFrame(list(range(num_exps)), columns = ['num_exp'])
init_df = exp_df.merge(empty_init_df, how = "cross")

for index, row in init_df.iterrows():

    init_df.at[index, 't_init'] = torch.randn(10)
    init_df.at[index, 'supp_init'] = rand_supp(10, 10)

real_data_DF = datafile_df.merge(hyperparam_df, how = "cross").merge(init_df, how = "cross")

experiment(real_data_DF)
real_data_DF['loss'] = real_data_DF['loss_values'].str[-1]
real_data_DF.to_pickle("real_data_DF.pkl")