import pandas as pd
from trop_fns import *
from WDDD_class import *
from experiments import *

sim_data_DF = make_DF(R_list = [50], NM_list = [[6,6],[10,6],[15,6],[10,10],[15,10], [15,15]], 
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

trained_scale_df = pd.read_pickle("trained_scale_df.pkl")
trained_lr_df = pd.read_pickle("trained_lr_df.pkl")

sim_data_DF = sim_data_DF.merge(trained_lr_df, on = ["N", "M"]).merge(trained_scale_df, on = ["N", "M"])

experiment(sim_data_DF, verbose_bool = False)
sim_data_DF.to_pickle("Data_Frames/sim_data_DF.pkl")