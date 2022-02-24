import pickle
import torch
import itertools

nn_params = {
    'layers_x': [16, 16],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,  # lambda x : x
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
    post_fix = f'_b={b}_D={D}_{N}'
    job_name = 'datasets/' + f'{dir_name}' + post_fix
    return job_name

def generate_parameters(job_dir,dir_name,bvec,Dvec,Nvec):
    num_list = itertools.product(bvec, Dvec, Nvec)
    methods=['baseline','doubly_robust']
    oracle_weights =[True,False]
    double_estimate_kme=[True,False]
    neural_cmes=[True,False]
    p_list = itertools.product(methods,oracle_weights,double_estimate_kme,neural_cmes)
    for (b,D,N) in num_list:
        data_dir = get_dir_name(dir_name,b,D,N)
        for (method,oracle_weight,de_kme,neural_cme) in p_list:

            job_name=f'ow={oracle_weight}_dek={de_kme}_ncme={neural_cme}'
            training_params = {'bs': 100,
                               'patience': 10,
                               'device': 'cuda:0',
                               'permute_e': True,
                               'permutations': 250,
                               'oracle_weights': oracle_weight,
                               'double_estimate_kme': de_kme,
                               'epochs': 0,
                               'debug_mode': False,
                               'neural_net_parameters': nn_parms_2,
                               'approximate_inverse': False,
                               'neural_cme': neural_cme
                               }
            experiment_params = {
                'experiment_save_path': f'{method}/{data_dir}_{job_name}_results', 'data_dir_load': f'{data_dir}',
                'num_exp': 100, 'nn_params': nn_params, 'training_params': training_params, 'cat_cols': [],
                'test_type': f'{method}', 'debug_mode': False
            }
            with  open(f'{job_dir}/{method}/{data_dir}_{job_name}.pickle', 'wb') as handle:
                pickle.dump(experiment_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    bvec = [0.0, 1e-3, 1e-2, 1e-2 * 2.5, 1e-2 * 5]
    Dvec = [5]
    Nvec = [2000]
    generate_parameters('large_mvp','somedir',bvec,Dvec,Nvec)

