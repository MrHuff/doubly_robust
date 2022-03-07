import numpy as np
import pandas as pd
# from zepid import load_sample_data, spline, RiskDifference
# from zepid.causal.gformula import TimeFixedGFormula, SurvivalGFormula
from zepid.causal.ipw import IPTW, IPMW
# from zepid.causal.snm import GEstimationSNM
# from zepid.causal.doublyrobust import AIPTW, TMLE

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import DomainWarning

class IPTW_pval(IPTW):
    def fit_pval(self, continuous_distribution='gaussian'):
        """Fit the specified marginal structural model using the calculated inverse probability of treatment weights.
               """
        if self._IPTW__mdenom is None: #wrangled member fuck this fucking retard who wrote this piece of shit
            raise ValueError('No model has been fit to generated predicted probabilities')
        if self.ms_model is None:
            raise ValueError('No marginal structural model has been specified')
        if self._miss_flag and not self._fit_missing_:
            warnings.warn("All missing outcome data is assumed to be missing completely at random. To relax this "
                          "assumption to outcome data is missing at random please use the `missing_model()` "
                          "function", UserWarning)

        ind = sm.cov_struct.Independence()
        full_msm = self.outcome + ' ~ ' + self.ms_model

        df = self.df.copy()
        if self.ipmw is None:
            if self._weight_ is None:
                df['_ipfw_'] = self.iptw
            else:
                df['_ipfw_'] = self.iptw * self.df[self._weight_]
        else:
            if self._weight_ is None:
                df['_ipfw_'] = self.iptw * self.ipmw
            else:
                df['_ipfw_'] = self.iptw * self.ipmw * self.df[self._weight_]
        df = df.dropna()

        if self._continuous_outcome:
            if (continuous_distribution == 'gaussian') or (continuous_distribution == 'normal'):
                f = sm.families.family.Gaussian()
            elif continuous_distribution == 'poisson':
                f = sm.families.family.Poisson()
            else:
                raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
            self._continuous_y_type = continuous_distribution
            fm = smf.gee(full_msm, df.index, df,
                         cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
            self.average_treatment_effect = pd.DataFrame()
            self.average_treatment_effect['labels'] = np.asarray(fm.params.index)
            self.average_treatment_effect.set_index(keys=['labels'], inplace=True)
            self.average_treatment_effect['ATE'] = np.asarray(fm.params)
            self.average_treatment_effect['SE(ATE)'] = np.asarray(fm.bse)
            self.average_treatment_effect['95%LCL'] = np.asarray(fm.conf_int()[0])
            self.average_treatment_effect['95%UCL'] = np.asarray(fm.conf_int()[1])
            self.D_pval = fm.pvalues[1].item()

        else:
            # Ignoring DomainWarnings from statsmodels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DomainWarning)

                # Estimating Risk Difference
                f = sm.families.family.Binomial(sm.families.links.identity())
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.risk_difference = pd.DataFrame()
                self.risk_difference['labels'] = np.asarray(fm.params.index)
                self.risk_difference.set_index(keys=['labels'], inplace=True)
                self.risk_difference['RD'] = np.asarray(fm.params)
                self.risk_difference['SE(RD)'] = np.asarray(fm.bse)
                self.risk_difference['95%LCL'] = np.asarray(fm.conf_int()[0])
                self.risk_difference['95%UCL'] = np.asarray(fm.conf_int()[1])

                # Estimating Risk Ratio
                f = sm.families.family.Binomial(sm.families.links.log())
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.risk_ratio = pd.DataFrame()
                self.risk_ratio['labels'] = np.asarray(fm.params.index)
                self.risk_ratio.set_index(keys=['labels'], inplace=True)
                self.risk_ratio['RR'] = np.exp(np.asarray(fm.params))
                self.risk_ratio['SE(log(RR))'] = np.asarray(fm.bse)
                self.risk_ratio['95%LCL'] = np.exp(np.asarray(fm.conf_int()[0]))
                self.risk_ratio['95%UCL'] = np.exp(np.asarray(fm.conf_int()[1]))

                # Estimating Odds Ratio
                f = sm.families.family.Binomial()
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.odds_ratio = pd.DataFrame()
                self.odds_ratio['labels'] = np.asarray(fm.params.index)
                self.odds_ratio.set_index(keys=['labels'], inplace=True)
                self.odds_ratio['OR'] = np.exp(np.asarray(fm.params))
                self.odds_ratio['SE(log(OR))'] = np.asarray(fm.bse)
                self.odds_ratio['95%LCL'] = np.exp(np.asarray(fm.conf_int()[0]))
                self.odds_ratio['95%UCL'] = np.exp(np.asarray(fm.conf_int()[1]))


class iptw_baseline_test():
    def __init__(self,X,T,Y,n_bootstraps):
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] + ['Y']+['D']
        self.cov_string =''
        for i in range(self.d):
            self.cov_string+=f' + x_{i}'


        self.dfs = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        self.n_bootstraps = n_bootstraps

        # iptw = IPTW(self.dfs, treatment='D')
        # iptw.treatment_model(self.cov_string,
        #                      print_results=False)
        #
        # iptw.marginal_structural_model('D')
        # iptw.fit()
        # self.ref_stat = iptw.average_treatment_effect.iloc[1, 0]
    def permutation_test(self):
        iptw = IPTW_pval(self.dfs, treatment='D',outcome='Y')
        iptw.treatment_model(self.cov_string,
                             print_results=False)
        iptw.marginal_structural_model('D')
        iptw.fit_pval()
        pval = iptw.D_pval
        self.ref_stat = iptw.average_treatment_effect.iloc[1, 0]
        return pval,self.ref_stat


