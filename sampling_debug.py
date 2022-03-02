import numpy as np
from tmle_baseline.vanilla_IPW import IPTW_pval
from data_generation.data_generation import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from zepid.causal.ipw import IPTW, IPMW
from zepid.causal.doublyrobust import AIPTW, TMLE
from vanilla_doublyrobust_baseline.vanilla_dr import doubly_robust
# from sklearn.ensemble import RandomForestClassifier
# from zepid.superlearner import GLMSL, StepwiseSL, SuperLearner
# from zepid.causal.doublyrobust import SingleCrossfitAIPTW, SingleCrossfitTMLE
# import statsmodels.api as sm


sns.set()

def debug_plot_weights(T,w_true):
    df = pd.DataFrame(np.concatenate([T,w_true],axis=1),columns=['T','prob'])
    sns.histplot(data=df,x='prob',hue='T',bins=50)
    plt.show()
def debug_plot_treatments(T,Y):
    df = pd.DataFrame(np.concatenate([T,Y],axis=1),columns=['T','Y'])
    sns.histplot(data=df,x='Y',hue='T',bins=50)
    plt.show()


if __name__ == '__main__':
    s=0
    D=5
    b=0.0
    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 35,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_1(**base_config)
    debug_plot_weights(T,w_true)
    debug_plot_treatments(T,Y)

    n, d = X.shape
    columns = [f'x_{i}' for i in range(d)] + ['Y'] + ['D']
    x_col = [f'x_{i}' for i in range(d)]
    cov_string = ''
    for i in range(d):
        cov_string += f' + x_{i}'

    dfs = pd.DataFrame(np.concatenate([X, Y, T], axis=1), columns=columns)
    n_bootstraps = 250

    print(doubly_robust(dfs,X=x_col,T='D',Y='Y'))

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

    # labels = ["LogR", "Step.int"]
    # candidates = [GLMSL(sm.families.family.Binomial()),
    #               StepwiseSL(sm.families.family.Binomial(), selection="forward", order_interaction=0),
    #               # RandomForestClassifier()
    #               ]
    #
    # sctmle = SingleCrossfitTMLE(dfs, exposure='D', outcome='Y')
    # sctmle.exposure_model(cov_string,
    #                       SuperLearner(candidates, labels, folds=10, loss_function="nloglik",verbose=True),
    #                       bound=0.01)
    # sctmle.outcome_model(cov_string,
    #                      SuperLearner(candidates, labels, folds=10, loss_function="nloglik",verbose=True))
    # sctmle.fit()
    #
    # sctmle.summary()

    #DONT PERMUTE THE E WEIGHTS!







