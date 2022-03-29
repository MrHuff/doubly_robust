
from doubly_robust_method.utils import *
from generate_job_parameters import *
import os
if __name__ == '__main__':
    #    methods=['doubly_robust_correct','baseline','doubly_robust']
    # dataset=  'distributions'
    fold = 'all_gpu_baselines_4'
    jobs = os.listdir(fold)
    jobs.sort()
    # print(jobs)
    # job = jobs[140]
    # experiment_params = {'experiment_save_path': f'all_gpu_real_results/doubly_robust_correct/{dataset}/ow=False_dek=True_ncme=False_tp=False_results',
    #                      'data_dir_load': f'datasets/{dataset}',
    #                      'num_exp': 100,
    #                      'nn_params': {'layers_x': [1], 'cat_size_list': [], 'dropout': 0.0, 'transformation': lambda x: x},
    #                      'training_params': {'bs': 100, 'patience': 10, 'device': 'cuda:0',
    #                                          'permute_e': True, 'permutations': 250,
    #                                          'oracle_weights': False,
    #                                          'double_estimate_kme': True,
    #                                          'epochs': True,
    #                                          'debug_mode': False,
    #                                          'neural_net_parameters': {'layers_x': [16, 8], 'cat_size_list': [], 'dropout': 0.0, 'transformation': torch.tanh, 'output_dim': 25},
    #                                          'approximate_inverse': False, 'neural_cme': False},
    #                      'cat_cols': [], 'test_type': 'doubly_robust_correct', 'debug_mode': False}
    for job in [jobs[1]]:
        experiment_params = load_obj(job,folder=f'{fold}/')
        print(experiment_params)
        c = experiment_object(**experiment_params)
        c.debug_mode=True
        c.run_experiments()
