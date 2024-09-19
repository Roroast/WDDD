import pandas as pd
from trop_fns import *
from WDDD_class import *
from experiments import *

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

supp_descent_DF = make_DF(R_list = [50], NM_list = [[6,6],[10,6],[15,6],[10,10],[15,10], [15,15]], 
                        type_list = ['branching', 'coalescent', 'gaussian'], 
                        grad_list = ["TA"], 
                        num_exps = 10, 
                        lr_df = None,
                        scale_df = None,
                        graph_list = ["comp", "incomp"],
                        p = [2],
                        steps = [40],
                        shift_num_steps = [50],
                        supp_num_steps = [50]
                       )

scales = list(np.logspace(-7, 2, num = 10, base = np.e))
lr_df = pd.DataFrame([np.e**(-1)], columns = ['lr'])
scale_df = pd.DataFrame(scales, columns = ['scale'])

supp_descent_DF = supp_descent_DF.merge(lr_df, how = "cross").merge(scale_df, how = "cross")

experiment(supp_descent_DF, verbose_bool = False)
supp_descent_DF['loss'] = supp_descent_DF['loss_values']
supp_descent_DF.to_pickle("DataFrames/supp_descent_DF.pkl")