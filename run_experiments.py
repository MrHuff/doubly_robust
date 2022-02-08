from doubly_robust_method.utils import *
import torch

if __name__ == '__main__':
    nn_params={
        'layers_x':[1],
        'dropout': 0.0,
          'transformation':lambda x : x
    }
    training_params={'bs':100,
                     'patience':10,
                     'device':'cuda:0',
                     'permute_e':False,
                     'permutations':250,
    }
    experiment_params={
        'experiment_save_path':'unit_test_dr' , 'data_dir_load':'unit_test_b=0.0', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
        'test_type':'doubly_robust', 'debug_mode':True
    }
    c=experiment_object(**experiment_params)
    c.run_experiments()

