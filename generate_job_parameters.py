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

def generate_parameters(job_dir,dir_name,bvec,Dvec,Nvec):
    num_list = list(itertools.product(bvec, Dvec, Nvec))
    # methods=['baseline','doubly_robust']
    # methods=['doubly_robust','doubleml']
    methods=['vanilla_dr','gformula','tmle','ipw','cf','bart']
    oracle_weights =[False]
    double_estimate_kme=[True]
    neural_cmes=[True]
    train_prop=[True]
    p_list = list(itertools.product(methods,oracle_weights,double_estimate_kme,neural_cmes,train_prop))

    if not os.path.exists(f'{job_dir}'):
        os.makedirs(f'{job_dir}')
    for (b,D,N) in num_list:
        data_dir,post_fix = get_dir_name(dir_name,b,D,N)
        data_indexing_string = dir_name+'_'+post_fix
        for (method,oracle_weight,de_kme,neural_cme,tp) in p_list:
            if method=='baseline':
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
                'experiment_save_path': f'{job_dir}_results/{method}/{dir_name}/{post_fix}_{job_name}_results', 'data_dir_load': f'{data_dir}',
                'num_exp': 2, 'nn_params': nn_params, 'training_params': training_params, 'cat_cols': [],
                'test_type': f'{method}', 'debug_mode': False
            }
            # print(experiment_params)


            with open(f'{job_dir}/{method}_{data_indexing_string}_{job_name}.pickle', 'wb') as handle:
                pickle.dump(experiment_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    # bvec=[0.0,0.01,0.025,0.05,0.1]
    # Dvec=[5]
    # Nvec=[100,250,500,1000,2000,5000]

    bvec=[0.0]
    Dvec=[5]
    Nvec=[100,5000]

    generate_parameters('baseline_test','unit_test',bvec,Dvec,Nvec)

