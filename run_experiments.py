from doubly_robust_method.utils import *
import torch

if __name__ == '__main__':
    #TODO: Neural network, figure cross validation/split strategy. Regression (i.e. E[Y|X]) under dependency and then under independency i.e. pre-permutation and post-permutation
    #Ok weirdness going on, kriks thing still working under useless weights.
    #Bellot cannot be directly applied, consider PDS instead.

    nn_params={
        'layers_x':[1],
        'dropout': 0.0,
          'transformation':lambda x : x
    }
    training_params={'bs':100,
                     'patience':10,
                     'device':'cuda:0',
                     'permute_e':True, #Don't think you are supposed to permute the weights...
                     'permutations':250,
                      'oracle_weights':False,
                     'double_estimate_kme': True,
                     'epochs':0
    }

    for b in [0.0,1e-3,1e-2,1e-1,0.5]:
        experiment_params={
            'experiment_save_path':f'dr_exp/dr_break_baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
            'test_type':'doubly_robust', 'debug_mode':False
        }
        c=experiment_object(**experiment_params)
        c.run_experiments()

