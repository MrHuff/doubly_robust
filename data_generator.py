from data_generation.data_generation import *
import os
import pickle
import itertools
if __name__ == '__main__':
    bvec=[0.0,0.01,0.025,0.05,0.1]
    Dvec=[5]
    Nvec=[100,250,500,1000,2000,5000]
    p_list = list(itertools.product(bvec,Dvec,Nvec))
    for (b,D,N) in p_list:
        post_fix = f'b={b}_D={D}_{N}'

        job_name = 'datasets/'+f'distributions/'+post_fix
        if not os.path.exists(job_name):
            os.makedirs(job_name)
        SEEDS=100
        config_list=[]
        #Issue is probably in weights estimation... When you have an extra double term things get dicey...
        for s in range(SEEDS):
            base_config = {'seed': s,
                           'ns': 1000,
                           'd': D,
                           'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                           # the thing just blows up regardless of what you do?!
                           # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                           'alpha_0': 0.05,  # 0.05,
                           'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.1,  # Confounding
                           'noise_var': 0.1,
                           'b': b
                           }
            config_list.append(base_config)
        for el in config_list:
            T, Y, X, w_true = case_distributional(**el)

            # T,Y,X,w_true=case_1(**el)
            s=el['seed']
            with open(f'{job_name}/job_{s}.pickle', 'wb')as handle:
                pickle.dump({'seed':s,'T':T,'Y':Y,'X':X,'W':w_true}, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # YY0,YY1=case_1_ref(**el)
            # print(YY0.squeeze()==(Y[T==0]))
            # print(YY1.squeeze()==(Y[T==1]))


"""
FIRST CASE - Outlier case

            base_config = {'seed': s,
                           'ns': N,
                           'd': D,
                           'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 35,
                           'alpha_0': 0.05,  # 0.05,
                           'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                           'noise_var': 0.1,
                           'b': b
                           }
"""


"""
SECOND CASE - Conditions satisfied

    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
"""

"""
THIRD CASE A - BANANA

    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.1,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_1_xy_banana(**base_config)

THIRD CASE B - SIN

    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 1.0,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_1_xy_sin(**base_config)

"""



"""
FORTH CASE - Robin suggestion:

    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_1_robin(**base_config)


"""

"""
FiFTH CASE - distributional differences 


    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.2,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_distributional(**base_config)


"""




