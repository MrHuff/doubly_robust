import copy

from data_generation.data_generation import *
import os
import pickle
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def debug_plot_weights(savedir,plt_name,T,w_true):
    df = pd.DataFrame(np.concatenate([T,w_true],axis=1),columns=['T','prob'])
    g=sns.histplot(data=df,x='prob',hue='T',bins=50)
    # g._legend.remove()
    plt.legend('', frameon=False)

    plt.savefig(f'{savedir}/{plt_name}_weights.png',bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.clf()

def debug_plot_treatments(savedir,plt_name,T,Y):
    df = pd.DataFrame(np.concatenate([T,Y],axis=1),columns=['T','Y'])
    g=sns.histplot(data=df,x='Y',hue='T',bins=50)
    # g._legend.remove()
    plt.legend('', frameon=False)

    plt.savefig(f'{savedir}/{plt_name}_y.png',bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.clf()

if __name__ == '__main__':
    if not os.path.exists('data_plot_dir_strong'):
        os.makedirs('data_plot_dir_strong')
    bvec=[0.0,0.1]
    Dvec=[5]
    Nvec=[500,5000]
    p_list = list(itertools.product(bvec,Dvec,Nvec))

    gen_list = [
        case_1,
        case_distributional,
        case_1,
        case_1_xy_banana,
        case_1_xy_sin,
        case_1_robin,
        case_distributional,
        case_distributional_2,
        case_distributional_3,
        case_break_weights
    ]
    base_list=[
        {'seed': 0,
         'ns': 0,
         'd': 0,
         'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 35,
         'alpha_0': 0.05,  # 0.05,
         'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 3,  # Confounding
         'noise_var': 0.1,
         'b': 0
         },
          {'seed': 0,
           'ns': 0,
           'd': 0,
           'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 10,
           'alpha_0': 0.05,  # 0.05,
           'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 3,  # Confounding
           'noise_var': 0.1,
           'b': 0
           },

    ] + [        {'seed': 0,
         'ns': 0,
         'd': 0,
         'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
         # the thing just blows up regardless of what you do?!
         # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
         'alpha_0': 0.05,  # 0.05,
         'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 3,  # Confounding
         'noise_var': 0.1,
         'b': 0
         }]*7 + [
        {'seed': 0,
                   'ns': 5000,
                   'd': 0,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 7.5,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': -1.5,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 3,  # Confounding
                   'noise_var': 0.1,
                   'b': 0
                   }]
    name_list =['unit_test',
            'distributions_middle_ground',
            'conditions_satisfied',
          'banana',
          'sin',
          'robin',
        'distributions',
        'distributions_uniform',
        'distributions_gamma',
        'nonlinear_treatment',
                ]

    for (generator_function,base_config,name) in zip(gen_list,base_list,name_list):
        for (b,D,N) in p_list:
            if name in ['distributions_middle_ground','distributions','distributions_uniform','distributions_gamma']:
                b=b*10
            post_fix = f'b={b}_D={D}_{N}'
            job_name = 'datasets/'+f'{name}_strong/'+post_fix
            if not os.path.exists(job_name):
                os.makedirs(job_name)
            SEEDS=1
            config_list=[]
            #Issue is probably in weights estimation... When you have an extra double term things get dicey...
            base_config['d'] = D
            base_config['b'] = b
            base_config['ns'] = N
            for s in range(SEEDS):
                base_config['seed']=s
                cp = copy.deepcopy(base_config)
                config_list.append(cp)
            for el in config_list:
                T, Y, X, w_true = generator_function(**el)
                s=el['seed']
            b = str(b).replace('.','')
            debug_plot_weights(savedir='data_plot_dir_strong',plt_name=f'{name}_{b}',T=T,w_true=w_true)
            debug_plot_treatments(savedir='data_plot_dir_strong',plt_name=f'{name}_{b}',T=T,Y=Y)

    display_these_datasets = [
        'conditions_satisfied',
        'robin',
        'distributions',
        'distributions_uniform',
        'distributions_gamma',
        ]





