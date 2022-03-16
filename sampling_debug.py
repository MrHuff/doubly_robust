import numpy as np
from data_generation.data_generation import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from vanilla_doublyrobust_baseline.vanilla_dr import *
from tmle_baseline.g_formula import *
from tmle_baseline.tmle_baseline import *
from tmle_baseline.vanilla_IPW import *
from CausalForest.causal_forest import *
from BART_baseline.BART import *
from WMMD.WMMD import WMMDTest
sns.set()

def debug_plot_weights(T,w_true):
    df = pd.DataFrame(np.concatenate([T,w_true],axis=1),columns=['T','prob'])
    sns.histplot(data=df,x='prob',hue='T',bins=50)
    plt.show()
def debug_plot_treatments(T,Y):
    df = pd.DataFrame(np.concatenate([T,Y],axis=1),columns=['T','Y'])
    sns.histplot(data=df,x='Y',hue='T',bins=50)
    plt.show()

def debug_different_models(X,T,Y):
    c_1 = vanilla_dr_baseline_test(X,T,Y,n_bootstraps=250)
    pval,stat=c_1.permutation_test()
    print(pval,stat)
    c_2 = gformula_baseline_test(X, T, Y, n_bootstraps=250)
    pval, stat = c_2.permutation_test()
    print(pval,stat)
    #
    #
    c_3 = iptw_baseline_test(X, T, Y, n_bootstraps=250)
    pval, stat = c_3.permutation_test()
    print(pval,stat)
    #
    c_4 = tmle_baseline_test(X, T, Y, n_bootstraps=250)
    pval, stat = c_4.permutation_test()
    print(pval,stat)


    c_6 = BART_baseline_test(X, T, Y,bootstrap=250)
    pval, stat = c_6.permutation_test()
    print(pval,stat)



    c7=WMMDTest(X,T,Y,device='cuda:0',n_permute=250)
    pval,stat=c7.permutation_test()
    print(pval,stat)
if __name__ == '__main__':
    s=0
    D=5
    b=1.0
    base_config = {'seed': 0,
         'ns': 5000,
         'd': 5,
         'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
         # the thing just blows up regardless of what you do?!
         # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
         'alpha_0': 0.05,  # 0.05,
         'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
         'noise_var': 0.1,
         'b': b
         }
    T, Y, X, w_true = case_distributional_3(**base_config)
    debug_plot_weights(T,w_true)
    debug_plot_treatments(T,Y)
    # debug_different_models(X,T,Y)
    # print(T.shape)
    # print(X.shape)
    # print(Y.shape)
    # print(w_true.shape)
    # n, d = X.shape
    # columns = [f'x_{i}' for i in range(d)] + ['Y'] + ['D']
    # x_col = [f'x_{i}' for i in range(d)]
    # cov_string = ''
    # for i in range(d):
    #     cov_string += f' + x_{i}'
    #
    # dfs = pd.DataFrame(np.concatenate([X, Y, T], axis=1), columns=columns)
    # n_bootstraps = 250
    #
    # debug_different_models(X,T,Y)

    # print(doubly_robust(dfs,X=x_col,T='D',Y='Y'))

    # g = TimeFixedGFormula(dfs, exposure='D', outcome='Y')
    # g.outcome_model(model='D' + cov_string,
    #                 print_results=False)
    # # Estimating marginal effect under treat-all plan
    # g.fit(treatment='all')
    # r_all = g.marginal_outcome
    #
    # # Estimating marginal effect under treat-none plan
    # g.fit(treatment='none')
    # r_none = g.marginal_outcome
    #
    # print(r_none,r_all)
    #
    # ref_stat = r_all - r_none
    # print(ref_stat)

    # iptw = IPTW_pval(dfs, treatment='D',outcome='Y')
    # iptw.treatment_model(cov_string,
    #                      print_results=False)
    #
    # iptw.marginal_structural_model('D')
    # iptw.fit_pval()
    # iptw.summary()
    # print(iptw.average_treatment_effect.iloc[1,0])

    # tmle = TMLE(dfs, exposure='D', outcome='Y')
    # tmle.exposure_model(cov_string)
    # tmle.missing_model(cov_string+' + D')
    # tmle.outcome_model('D' + cov_string)
    # tmle.fit()
    # tmle.summary()

    #DONT PERMUTE THE E WEIGHTS!







