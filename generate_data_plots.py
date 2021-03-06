import copy
from data_generation.data_generation import *
import os
import pickle
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from post_process_job_plots import *
sns.set()
cp = sns.color_palette()
print(cp)
def debug_plot_weights(savedir,plt_name,T,w_true):
    df = pd.DataFrame(np.concatenate([T,w_true],axis=1),columns=['T','prob'])
    g=sns.histplot(data=df,x='prob',hue='T',bins=50)
    L = plt.legend()
    L.get_texts()
    print(L)
    plt.legend('', frameon=False)
    g.set_ylabel('Count',fontsize=20)
    g.set_xlabel('Propensity score',fontsize=20)
    g.set_yticklabels([int(f) for f in g.get_yticks()], size=15)
    g.set_xticklabels([round(f,1) for f in g.get_xticks()], size=15)

    # print(g)
    # g._legend.remove()
    plt.savefig(f'{savedir}/{plt_name}_weights.png',bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.clf()
def debug_plot_treatments(savedir,plt_name,T,Y):
    df = pd.DataFrame(np.concatenate([T,Y],axis=1),columns=['T','Y'])
    g=sns.histplot(data=df,x='Y',hue='T',bins=50)
    # print(g)
    L = plt.legend()
    L.get_texts()
    print(L)
    plt.legend('', frameon=False)
    # g._legend.remove()
    g.set_ylabel('Count',fontsize=20)
    g.set_xlabel('Y',fontsize=20)
    g.set_yticklabels([int(f) for f in g.get_yticks()], size=15)
    g.set_xticklabels([round(f,1) for f in g.get_xticks()], size=15)

    plt.savefig(f'{savedir}/{plt_name}_y.png',bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.clf()


bvec = [0.0, 0.1]
Dvec = [5]
Nvec = [5000]
p_list = list(itertools.product(bvec, Dvec, Nvec))

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
base_list = [
                {'seed': 0,
                 'ns': 0,
                 'd': 0,
                 'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 35,
                 'alpha_0': 0.05,  # 0.05,
                 'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                 'noise_var': 0.1,
                 'b': 0
                 },
                {'seed': 0,
                 'ns': 0,
                 'd': 0,
                 'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 10,
                 'alpha_0': 0.05,  # 0.05,
                 'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                 'noise_var': 0.1,
                 'b': 0
                 },

            ] + [{'seed': 0,
                  'ns': 0,
                  'd': 0,
                  'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
                  # the thing just blows up regardless of what you do?!
                  # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                  'alpha_0': 0.05,  # 0.05,
                  'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                  'noise_var': 0.1,
                  'b': 0
                  }] * 7 + [
                {'seed': 0,
                 'ns': 5000,
                 'd': 0,
                 'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 7.5,  # Treatment assignment
                 # the thing just blows up regardless of what you do?!
                 # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                 'alpha_0': -1.5,  # 0.05,
                 'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                 'noise_var': 0.1,
                 'b': 0
                 }]
name_list = [
            'unit_test',
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
if __name__ == '__main__':
    if not os.path.exists('data_plot_dir'):
        os.makedirs('data_plot_dir')

    for (generator_function, base_config, name) in zip(gen_list, base_list, name_list):
        for (b, D, N) in p_list:
            if name in ['distributions_middle_ground', 'distributions', 'distributions_uniform', 'distributions_gamma']:
                b = b * 10
            post_fix = f'b={b}_D={D}_{N}'
            job_name = 'datasets/' + f'{name}/' + post_fix
            if not os.path.exists(job_name):
                os.makedirs(job_name)
            SEEDS = 1
            config_list = []
            # Issue is probably in weights estimation... When you have an extra double term things get dicey...
            base_config['d'] = D
            base_config['b'] = b
            base_config['ns'] = N
            for s in range(SEEDS):
                base_config['seed'] = s
                cp = copy.deepcopy(base_config)
                config_list.append(cp)
            for el in config_list:
                T, Y, X, w_true = generator_function(**el)
            b = str(b).replace('.','')
            debug_plot_weights(savedir='data_plot_dir',plt_name=f'{name}_{b}',T=T,w_true=w_true)
            debug_plot_treatments(savedir='data_plot_dir',plt_name=f'{name}_{b}',T=T,Y=Y)








