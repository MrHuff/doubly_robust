from doubly_robust_method.utils import *
import torch

def baseline_run(b_list,nn_params):

    for b in b_list:
        training_params = {'bs': 100,
                           'patience': 10,
                           'device': 'cuda:0',
                           'permute_e': True,  # Don't think you are supposed to permute the weights...
                           'permutations': 250,
                           'oracle_weights': False,
                           'double_estimate_kme': True,
                           'epochs': 0
                           }
        experiment_params={
            'experiment_save_path':f'baseline/break_baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
            'test_type':'baseline', 'debug_mode':False
        }
        c=experiment_object(**experiment_params)
        c.run_experiments()

        # training_params = {'bs': 100,
        #                    'patience': 10,
        #                    'device': 'cuda:0',
        #                    'permute_e': True,  # Don't think you are supposed to permute the weights...
        #                    'permutations': 250,
        #                    'oracle_weights': False,
        #                    'double_estimate_kme': True,
        #                    'epochs': 100
        #                    }
        # experiment_params={
        #     'experiment_save_path':f'baseline/baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
        #     'test_type':'baseline', 'debug_mode':False
        # }
        # c=experiment_object(**experiment_params)
        # c.run_experiments()

        training_params = {'bs': 100,
                           'patience': 10,
                           'device': 'cuda:0',
                           'permute_e': True,  # Don't think you are supposed to permute the weights...
                           'permutations': 250,
                           'oracle_weights': True,
                           'double_estimate_kme': True,
                           'epochs': 100
                           }
        experiment_params={
            'experiment_save_path':f'baseline/oracle_baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
            'test_type':'baseline', 'debug_mode':False
        }
        c=experiment_object(**experiment_params)
        c.run_experiments()

def dr_run(b_list,nn_params):
    nn_parms_2 = {
        'layers_x': [16,8],
        'cat_size_list':[],
        'dropout': 0.3,
        'transformation': torch.tanh,
        'output_dim': 10,
    }
    #Uppgrade conditional expectation estimate.
    training_params = {'bs': 100,
                       'patience': 10,
                       'device': 'cuda:0',
                       'permute_e': True,  # Don't think you are supposed to permute the weights...
                       'permutations': 250,
                       'oracle_weights': False,
                       'double_estimate_kme': True,
                       'epochs': 0,
                       'debug_mode': False,
                       'neural_net_parameters':nn_parms_2,
                       'approximate_inverse':False,
                       'neural_cme':True
                       }
    for b in b_list:
        # training_params['neural_cme']=False
        experiment_params={
            'experiment_save_path':f'dr_exp/neural_ablation_baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
            'test_type':'doubly_robust', 'debug_mode':True
        }
        c=experiment_object(**experiment_params)
        c.run_experiments()
        # training_params['neural_cme']=False
        # experiment_params={
        #     'experiment_save_path':f'dr_exp/ablation_baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
        #     'test_type':'doubly_robust', 'debug_mode':True
        # }
        # c=experiment_object(**experiment_params)
        # c.run_experiments()
        # training_params['epochs']=100
        # experiment_params={
        #     'experiment_save_path':f'baseline/baseline_{b}' , 'data_dir_load':f'unit_test_b={b}', 'num_exp':100, 'nn_params':nn_params, 'training_params':training_params, 'cat_cols':[],
        #     'test_type':'baseline', 'debug_mode':False
        # }
        # c=experiment_object(**experiment_params)
        # c.run_experiments()

if __name__ == '__main__':

    #TODO: Neural network, figure cross validation/split strategy. Regression (i.e. E[Y|X]) under dependency and then under independency i.e. pre-permutation and post-permutation
    #Ok weirdness going on, kriks thing still working under useless weights.
    #Bellot cannot be directly applied, consider PDS instead.
    nn_params={
        'layers_x':[1],
        'cat_size_list': [],
        'dropout': 0.0,
          'transformation':lambda x : x
    }
    # b_list=[0.0,1e-3,1e-2,1e-2*2.5,1e-2*5]
    b_list=[0.0]
    # baseline_run(b_list,nn_params)
    dr_run(b_list,nn_params)



