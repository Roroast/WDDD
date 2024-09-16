import pandas as pd
from trop_fns import *
from WDDD_class import *
from experiments import *

sim_datasets = []
for N in [6,10,15]:
    for data_type in ['Norm', 'Coal', 'Gaus']:
        X = np.load(f"Data/{data_type}{N}.0_R50.0.npy")
        sim_datasets.append(X)
        
trained_scale_df = pd.read_pickle("trained_scale_df.pkl")

sim_comp_DF = make_DF(R_list = [50], NM_list = [[6,6],[10,6],[15,6],[10,10],[15,10], [15,15]], 
                        type_list = ['branching', 'coalescent', 'gaussian'], 
                        grad_list = ["TA"], 
                        num_exps = 10, 
                        lr_df = None,
                        scale_df = None,
                        graph_list = ["incomp"],
                        p = [2],
                        steps = [40],
                        shift_num_steps = [50],
                        supp_num_steps = [50],
                        normalised_data = True
                       )

lr_df = pd.DataFrame([np.e**(-1)], columns = ['lr'])
scale_df = pd.DataFrame([np.e**(-2)], columns = ['scale'])

sim_comp_DF = sim_comp_DF.merge(lr_df, how = "cross").merge(scale_df, how = "cross")

experiment(sim_comp_DF)
sim_comp_DF['loss'] = sim_comp_DF['loss_values'].str[-1]
sim_comp_DF.to_pickle("sim_comp_DF.pkl")