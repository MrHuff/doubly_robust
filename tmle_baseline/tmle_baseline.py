import numpy as np
import pandas as pd
from zepid.causal.doublyrobust import AIPTW, TMLE

from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric


class tmle_baseline_test():
    def __init__(self,X,T,Y,n_bootstraps):
        X=X[:, ~(X == X[0, :]).all(0)]
        X = X[:,:25] #prevent numerical overflows
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] + ['Y']+['D']
        self.cov_string =''
        for i in range(self.d):
            self.cov_string+=f' + x_{i}'
        l = np.unique(Y)
        # if len(l)<4:
        Y = np.exp(Y)
        self.dfs = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        self.columns = columns
        self.n_bootstraps = n_bootstraps
        self.X,self.T,self.Y = X,T,Y
        tmle = TMLE(self.dfs, exposure='D', outcome='Y')
        tmle.exposure_model(self.cov_string, print_results=False)
        tmle.outcome_model('D'+self.cov_string, print_results=False)
        tmle.fit()
        # tmle.summary()


        self.ref_stat = tmle.average_treatment_effect
    def permutation_test(self):
        rd_results = []
        for i in range(self.n_bootstraps):
            Y = np.random.permutation(self.Y)
            s = pd.DataFrame(np.concatenate([self.X,Y,self.T],axis=1),columns=self.columns)
            # s = self.dfs.sample(n=self.n, replace=True)
            tmle = TMLE(s, exposure='D', outcome='Y')
            tmle.exposure_model(self.cov_string, print_results=False)
            tmle.outcome_model('D' + self.cov_string, print_results=False)
            tmle.fit()
            rd_results.append(tmle.average_treatment_effect)
        rd_results = np.array(rd_results)
        pval=calculate_pval_symmetric(rd_results,self.ref_stat)

        return pval,self.ref_stat


