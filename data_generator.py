from data_generation.data_generation import *
import os
import pickle
if __name__ == '__main__':
    for b in [0.0,1e-3,1e-2,1e-1,0.5]:
        job_name = f'unit_test_b={b}'
        if not os.path.exists(job_name):
            os.makedirs(job_name)
        SEEDS=100
        config_list=[]
        for s in range(SEEDS):
            base_config={'seed':s,
                         'ns':500,
                         'd':5,
                         'alpha_vec':np.random.randn(1,5),
                         'alpha_0':np.random.randn(),
                         'beta_vec':np.random.randn(5),
                         'noise_var':np.random.rand(),
                         'b':2.
            }
            config_list.append(base_config)
        for el in config_list:
            T,Y,X=case_1(**el)
            s=el['seed']
            with  open(f'{job_name}/job_{s}.pickle', 'wb')as handle:
                pickle.dump({'seed':s,'T':T,'Y':Y,'X':X}, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # YY0,YY1=case_1_ref(**el)
            # print(YY0.squeeze()==(Y[T==0]))
            # print(YY1.squeeze()==(Y[T==1]))







