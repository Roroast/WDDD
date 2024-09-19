import pandas as pd
from trop_fns import *
from WDDD_class import *
from experiments import *

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

shift_optimisation_DF = make_DF(R_list = [50], NM_list = [[6,6],[10,6],[15,6],[10,10],[15,10], [15,15]], 
                        type_list = ['branching', 'coalescent', 'gaussian'], 
                        grad_list = ["CD", "TD", "CA", "TA"], 
                        num_exps = 10, 
                        lr_df = None,
                        scale_df = None,
                        graph_list = ["comp"],
                        p = [2],
                        steps = [40],
                        shift_num_steps = [50],
                        supp_num_steps = [50]
                       )

lrs = list(np.logspace(-6, 4, num = 11, base = np.e))
lr_df = pd.DataFrame(lrs, columns = ['lr'])
scale_df = pd.DataFrame([0.1], columns = ['scale'])

shift_optimisation_DF = shift_optimisation_DF.merge(lr_df, how = "cross").merge(scale_df, how = "cross")

experiment(shift_optimisation_DF, verbose_bool = False)
shift_optimisation_DF['loss'] = shift_optimisation_DF['loss_values']
shift_optimisation_DF.to_pickle("DataFrames/shift_optimisation_DF.pkl")