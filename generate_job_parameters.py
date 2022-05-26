import os.path
import pickle
import torch
import itertools

def linear(x):
    return x

nn_params = {
    'layers_x': [1],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': linear
}
nn_params_class = {
    'layers_x': [16, 8],
    'cat_size_list': [],
    'dropout': 0.1,
    'transformation': torch.tanh,
    'output_dim': 1,
}

nn_parms_2 = {
    'layers_x': [16, 8],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,
    'output_dim': 25,
}

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def get_dir_name(dir_name,b,D,N):
    post_fix = f'b={b}_D={D}_{N}'
    job_name = 'datasets/' + f'{dir_name}/' + post_fix
    return job_name,post_fix

def get_dir_name_type_1(dir_name,b,D,N):
    post_fix = f'beta={b}_D={D}_{N}'
    job_name = 'datasets/' + f'{dir_name}/' + post_fix
    return job_name,post_fix

def generate_parameters_type_1(job_dir,dir_names,bvec,Dvec,Nvec,methods,nn_params):
    num_list = list(itertools.product(bvec, Dvec, Nvec))
    oracle_weights =[True]
    double_estimate_kme=[False]
    neural_cmes=[False]
    train_prop=[False]
    p_list = list(itertools.product(methods,oracle_weights,double_estimate_kme,neural_cmes,train_prop))

    if not os.path.exists(f'{job_dir}'):
        os.makedirs(f'{job_dir}')
    for dir_name in dir_names:

        for (method, oracle_weight, de_kme, neural_cme, tp) in p_list:
            for (b,D,N) in num_list:
                if method in ['baseline', 'baseline_correct']:
                    de_kme=False
                    neural_cme=False
                if oracle_weight:
                    tp=False
                if method in ['doubleml','vanilla_dr','gformula','tmle','ipw','cf','bart','wmmd']:
                    de_kme=False
                    neural_cme=False
                    tp=True
                    oracle_weight=False
                    if method=='wmmd':
                        N=min(N,500)

                data_dir, post_fix = get_dir_name_type_1(dir_name, b, D, N)
                data_indexing_string = dir_name + '_' + post_fix

                job_name=f'ow={oracle_weight}_dek={de_kme}_ncme={neural_cme}_tp={tp}'
                training_params = {'bs': 100,
                                   'patience': 10,
                                   'device': 'cuda:0',
                                   'permute_e': True,
                                   'permutations': 250,
                                   'oracle_weights': oracle_weight,
                                   'double_estimate_kme': de_kme,
                                   'epochs': 100 if tp else 0,
                                   'debug_mode': False,
                                   'neural_net_parameters': nn_parms_2,
                                   'approximate_inverse': False,
                                   'neural_cme': neural_cme
                                   }
                experiment_params = {
                    'experiment_save_path': f'{job_dir}_results/{method}/{dir_name}/{post_fix}_{job_name}_results', 'data_dir_load': f'{data_dir}',
                    'num_exp': 100, 'nn_params': nn_params, 'training_params': training_params, 'cat_cols': [],
                    'test_type': f'{method}', 'debug_mode': False
                }

                with open(f'{job_dir}/{dir_name}_{method}_{data_indexing_string}_{job_name}.pickle', 'wb') as handle:
                    pickle.dump(experiment_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_parameters(job_dir,dir_names,bvec,Dvec,Nvec,methods,nn_params):
    num_list = list(itertools.product(bvec, Dvec, Nvec))
    oracle_weights =[False]
    double_estimate_kme=[True]
    neural_cmes=[False]
    train_prop=[True]
    p_list = list(itertools.product(methods,oracle_weights,double_estimate_kme,neural_cmes,train_prop))

    if not os.path.exists(f'{job_dir}'):
        os.makedirs(f'{job_dir}')


    for dir_name in dir_names:
        for (method, oracle_weight, de_kme, neural_cme, tp) in p_list:
            for (b,D,N) in num_list:
                if 'strong_two' in dir_name:
                    b=b*10
                if method in ['baseline', 'baseline_correct']:
                    de_kme=False
                    neural_cme=False
                if oracle_weight:
                    tp=False
                if method in ['doubleml','vanilla_dr','gformula','tmle','ipw','cf','bart','wmmd']:
                    de_kme=False
                    neural_cme=False
                    tp=True
                    oracle_weight=False
                    if method=='wmmd':
                        N=min(N,500)

                data_dir, post_fix = get_dir_name(dir_name, b, D, N)
                data_indexing_string = dir_name + '_' + post_fix

                job_name=f'ow={oracle_weight}_dek={de_kme}_ncme={neural_cme}_tp={tp}'
                training_params = {'bs': 100,
                                   'patience': 10,
                                   'device': 'cuda:0',
                                   'permute_e': True,
                                   'permutations': 250,
                                   'oracle_weights': oracle_weight,
                                   'double_estimate_kme': de_kme,
                                   'epochs': 100 if tp else 0,
                                   'debug_mode': False,
                                   'neural_net_parameters': nn_parms_2,
                                   'approximate_inverse': False,
                                   'neural_cme': neural_cme
                                   }
                experiment_params = {
                    'experiment_save_path': f'{job_dir}_results/{method}/{dir_name}/{post_fix}_{job_name}_results', 'data_dir_load': f'{data_dir}',
                    'num_exp': 100, 'nn_params': nn_params, 'training_params': training_params, 'cat_cols': [],
                    'test_type': f'{method}', 'debug_mode': False
                }
                # print(experiment_params)


                with open(f'{job_dir}/{dir_name}_{method}_{data_indexing_string}_{job_name}.pickle', 'wb') as handle:
                    pickle.dump(experiment_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_parameters_real_datasets(job_dir,dir_names,methods,nn_params,ds_dir='datasets'):
    # methods=['doubly_robust','baseline']
    # methods=['doubly_robust','doubleml']
    # methods=['doubleml','vanilla_dr','gformula','tmle','ipw','cf','bart']
    oracle_weights =[False]
    double_estimate_kme=[True]
    neural_cmes=[False]
    train_prop=[True]
    p_list = list(itertools.product(methods,oracle_weights,double_estimate_kme,neural_cmes,train_prop))

    if not os.path.exists(f'{job_dir}'):
        os.makedirs(f'{job_dir}')

    for dir_name in dir_names:

        for (method,oracle_weight,de_kme,neural_cme,tp) in p_list:
            if method in ['baseline','baseline_correct']:
                de_kme=False
                neural_cme=False
            if oracle_weight:
                tp=False
            if method in ['doubleml','vanilla_dr','gformula','tmle','ipw','cf','bart']:
                de_kme=False
                neural_cme=False
                tp=True
                oracle_weight=False

            job_name=f'ow={oracle_weight}_dek={de_kme}_ncme={neural_cme}_tp={tp}'
            training_params = {'bs': 100,
                               'patience': 10,
                               'device': 'cuda:0',
                               'permute_e': True,
                               'permutations': 250,
                               'oracle_weights': oracle_weight,
                               'double_estimate_kme': de_kme,
                               'epochs': 100 if tp else 0,
                               'debug_mode': False,
                               'neural_net_parameters': nn_parms_2,
                               'approximate_inverse': False,
                               'neural_cme': neural_cme
                               }
            experiment_params = {
                'experiment_save_path': f'{job_dir}_results/{method}/{dir_name}/{job_name}_results', 'data_dir_load': f'{ds_dir}/{dir_name}',
                'num_exp': 100, 'nn_params': nn_params, 'training_params': training_params, 'cat_cols': [],
                'test_type': f'{method}', 'debug_mode': False
            }
            # print(experiment_params)

            with open(f'{job_dir}/{dir_name}_{method}_{job_name}.pickle', 'wb') as handle:
                pickle.dump(experiment_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



def generate_all_cpu_baselines():
    bvec=[0.0,0.01,0.025,0.05,0.1]
    Dvec=[5]
    Nvec=[500,5000]
    dsl = [
          'conditions_satisfied',
          'robin',
        'distributions',
        'distributions_uniform',
        'distributions_gamma',
                ]
    ds = [el+'_strong_two' for el in dsl]
    # ds = dsl+[el+'_strong_2' for el in dsl]

    ds_real = [
        'twins_2500',
        'twins_2500_null',
        'lalonde_100',
        'lalonde_100_null',
    ]
    methods = ['vanilla_dr', 'gformula', 'tmle', 'ipw', 'doubleml','cf','bart']
    generate_parameters(f'all_cpu_baselines_strong_2',ds,bvec,Dvec,Nvec,methods=methods,nn_params=nn_params)
    # generate_parameters_real_datasets(f'all_cpu_real',ds_real,methods,nn_params=nn_params,ds_dir='real_cpu')

def generate_all_gpu_baselines():
    if not os.path.exists('all_gpu_baselines_2'):
        os.makedirs('all_gpu_baselines_2')
    if not os.path.exists('all_gpu_real'):
        os.makedirs('all_gpu_real')

    bvec=[0.0,0.01,0.025,0.05,0.1]
    Dvec=[5]
    Nvec=[500,5000]
    dsl = [

          'conditions_satisfied',
          'robin',
        'distributions',
        'distributions_uniform',
        'distributions_gamma',
        'nonlinear_treatment',
                ]
    ds = [el+'_strong_two' for el in dsl]
    # ds = dsl+[el+'_strong_2' for el in dsl]
    ds_real = [
        'twins_2500',
        'twins_2500_null',
        'lalonde_100',
        'lalonde_100_null',
        'inspire_1000',
        'inspire_1000_null',
    ]
    methods=['doubly_robust_correct']
    generate_parameters_real_datasets(f'all_gpu_real_fix_table',ds_real,methods,nn_params)
    # methods = ['doubly_robust_correct','wmmd','baseline','baseline_correct']
    methods = ['doubly_robust_correct','wmmd']
    generate_parameters(f'all_gpu_baselines_strong_2',ds,bvec,Dvec,Nvec,methods,nn_params)


    # methods=['doubly_robust_correct']
    # generate_parameters(f'all_gpu_baselines_3',ds,bvec,Dvec,Nvec,methods,nn_params)
    # methods=['doubly_robust_correct_sampling']
    # generate_parameters(f'all_gpu_baselines_4',ds,bvec,Dvec,Nvec,methods,nn_params)
    # methods=['doubly_robust_correct','baseline','doubly_robust','baseline_correct']
    generate_parameters_real_datasets(f'all_gpu_real',ds_real,methods,nn_params)
    #
    #
    #
    # bvec=[0.0,0.01,0.02,0.03,0.04,0.05,0.1,0.5,1.,2.,3.]
    bvec=[0.1,0.5,1.,2.,3.]
    Dvec=[5]
    Nvec=[5000]
    ds = [
          'conditions_satisfied_type_two',
        'distributions_type_two',
                ]
    methods=['doubly_robust_correct']
    generate_parameters_type_1(f'type_two',ds,bvec,Dvec,Nvec,methods,nn_params)

    bvec=[0.25]
    Dvec=[1]
    Nvec=[100,500,1000,2000,5000,7500,10000]
    ds = [
          'fisher_example',
                ]
    methods=['baseline_correct']
    generate_parameters_type_1(f'fisher_jobs',ds,bvec,Dvec,Nvec,methods,nn_params)


if __name__ == '__main__':
    generate_all_cpu_baselines()
    generate_all_gpu_baselines()

