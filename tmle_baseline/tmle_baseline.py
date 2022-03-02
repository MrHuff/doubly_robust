import numpy as np
import pandas as pd
from zepid.causal.doublyrobust import AIPTW, TMLE
from  doubly_robust_method.utils import testing_class



class tmle_baseline_test():
    def __init__(self,X,Y,T,n_bootstraps):
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] + ['Y']+['D']
        self.cov_string =''
        for i in range(self.d):
            self.cov_string+=f' + x_{i}'
        self.dfs = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        self.n_bootstraps = n_bootstraps


        tmle = TMLE(self.dfs, exposure='D', outcome='Y')
        tmle.exposure_model(self.cov_string)
        tmle.outcome_model('D'+self.cov_string)
        tmle.fit()
        tmle.summary()


        self.ref_stat = tmle.self.average_treatment_effect
    def permutation_test(self):
        rd_results = []
        for i in range(self.n_bootstraps):
            s = self.dfs.sample(n=self.n, replace=True)
            tmle = TMLE(s, exposure='D', outcome='Y')
            tmle.exposure_model(self.cov_string)
            tmle.outcome_model('D' + self.cov_string)
            tmle.fit()
            tmle.summary()
            rd_results.append(tmle.self.average_treatment_effect)
        rd_results = np.array(rd_results)
        pval=testing_class.calculate_pval_symmetric(rd_results,self.ref_stat)

        return pval,self.ref_stat


