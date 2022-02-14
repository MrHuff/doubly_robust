from data_generation.data_generation import *
import os
import pickle
if __name__ == '__main__':
    D=5
    for b in [0.0,1e-3,1e-2,1e-2*2.5,1e-2*5]:
        job_name = f'unit_test_b={b}'
        if not os.path.exists(job_name):
            os.makedirs(job_name)
        SEEDS=100
        config_list=[]
        for s in range(SEEDS):
            base_config={'seed':s,
                         'ns':1000,
                         'd':D,
                         'alpha_vec':np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                         'alpha_0':0.05,
                         'beta_vec':np.array([0.1,0.2,0.3,0.4,0.5]),#np.random.randn(5)*0.05,
                         'noise_var':0.1,
                         'b':b
            }
            config_list.append(base_config)
        for el in config_list:
            T,Y,X,w_true=case_1(**el)
            s=el['seed']
            with  open(f'{job_name}/job_{s}.pickle', 'wb')as handle:
                pickle.dump({'seed':s,'T':T,'Y':Y,'X':X,'W':w_true}, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # YY0,YY1=case_1_ref(**el)
            # print(YY0.squeeze()==(Y[T==0]))
            # print(YY1.squeeze()==(Y[T==1]))







