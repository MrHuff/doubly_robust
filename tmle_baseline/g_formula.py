import numpy as np
import pandas as pd
# from zepid import load_sample_data, spline, RiskDifference
from zepid.causal.gformula import TimeFixedGFormula, SurvivalGFormula
# from zepid.causal.ipw import IPTW, IPMW
# from zepid.causal.snm import GEstimationSNM
# from zepid.causal.doublyrobust import AIPTW, TMLE
from doubly_robust_method.utils import testing_class

class gformula_baseline_test():
    def __init__(self,X,Y,T,n_bootstraps):
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] + ['Y']+['D']
        self.cov_string =''
        for i in range(self.d):
            self.cov_string+=f' + x_{i}'


        self.dfs = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        self.n_bootstraps = n_bootstraps

        g = TimeFixedGFormula(self.dfs, exposure='D', outcome='Y')
        g.outcome_model(model='D' + self.cov_string,
                        print_results=False)
        # Estimating marginal effect under treat-all plan
        g.fit(treatment='all')
        r_all = g.marginal_outcome

        # Estimating marginal effect under treat-none plan
        g.fit(treatment='none')
        r_none = g.marginal_outcome

        self.ref_stat = r_all - r_none

    def permutation_test(self):
        rd_results = []
        for i in range(self.n_bootstraps):
            s = self.dfs.sample(n=self.n, replace=True)
            g = TimeFixedGFormula(s, exposure='D', outcome='Y')
            g.outcome_model(model='D'+self.cov_string,
                            print_results=False)
            g.fit(treatment='all')
            r_all = g.marginal_outcome
            g.fit(treatment='none')
            r_none = g.marginal_outcome
            rd_results.append(r_all - r_none)
        # self.dml_irm_obj.fit()
        # pval= self.dml_irm_obj.summary['P>|t|'].item()
        # stat= self.dml_irm_obj.summary['P>|t|'].item()
        rd_results = np.array(rd_results)
        pval=testing_class.calculate_pval_symmetric(rd_results,self.ref_stat)
        return pval,self.ref_stat


