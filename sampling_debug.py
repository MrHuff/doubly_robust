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
    # c_1 = vanilla_dr_baseline_test(X,T,Y,n_bootstraps=250)
    # pval,stat=c_1.permutation_test()
    # print(pval,stat)
    # c_2 = gformula_baseline_test(X, T, Y, n_bootstraps=250)
    # pval, stat = c_2.permutation_test()
    # print(pval,stat)
    #
    #
    # c_3 = iptw_baseline_test(X, T, Y, n_bootstraps=250)
    # pval, stat = c_3.permutation_test()
    # print(pval,stat)
    #
    # c_4 = tmle_baseline_test(X, T, Y, n_bootstraps=250)
    # pval, stat = c_4.permutation_test()
    # print(pval,stat)


    # c_6 = BART_baseline_test(X, T, Y,X,T,X,T,bootstrap=250)
    # pval, stat = c_6.permutation_test()
    # print(pval,stat)


    # c_5 = CausalForest_baseline_test(X, T, Y,X,X,bootstrap=250)
    # pval, stat = c_5.permutation_test()
    # print(pval,stat)

    c7=WMMDTest(X,T,Y,250)
    pval,stat=c7.permutation_test()
    print(pval,stat)
if __name__ == '__main__':
    s=0
    D=5
    b=0.0
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
    debug_plot_weights(T,w_true)
    debug_plot_treatments(T,Y)
    debug_different_models(X,T,Y)


    #DONT PERMUTE THE E WEIGHTS!







