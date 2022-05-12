import numpy as np
import pandas as pd
from zepid.causal.doublyrobust import AIPTW, TMLE
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import logistic, norm
from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric
from zepid.causal.doublyrobust.utils import tmle_unit_bounds, tmle_unit_unbound
from zepid.calc import probability_to_odds, odds_to_probability, probability_bounds
import warnings
class TMLE_pval(TMLE):
    def fit(self):
        """Calculate the effect measures from the predicted exposure probabilities and predicted outcome values using
               the TMLE procedure. Confidence intervals are calculated using influence curves.
               Note
               ----
               Exposure and outcome models must be specified prior to `fit()`
               Returns
               -------
               TMLE gains `risk_difference`, `risk_ratio`, and `odds_ratio` for binary outcomes and
               `average _treatment_effect` for continuous outcomes
               """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')
        if self._miss_flag and not self._fit_missing_model:
            warnings.warn("No missing data model has been specified. All missing outcome data is assumed to be "
                          "missing completely at random. To relax this assumption to outcome data is missing at random"
                          "please use the `missing_model()` function", UserWarning)

        # Step 4) Calculating clever covariate (HAW)
        if self._miss_flag and self._fit_missing_model:
            self.g1W_total = self.g1W * self.m1W
            self.g0W_total = self.g0W * self.m0W
        else:
            self.g1W_total = self.g1W
            self.g0W_total = self.g0W
        H1W = self.df[self.exposure] / self.g1W_total
        H0W = -(1 - self.df[self.exposure]) / self.g0W_total
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self.outcome]
        log = sm.GLM(y, np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)),
                     family=f, missing='drop').fit()
        self._epsilon = log.params
        Qstar1 = logistic.cdf(np.log(probability_to_odds(self.QA1W)) + self._epsilon[0] / self.g1W_total)
        Qstar0 = logistic.cdf(np.log(probability_to_odds(self.QA0W)) - self._epsilon[1] / self.g0W_total)
        Qstar = log.predict(np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)))

        # Step 6) Calculating Psi
        if self.alpha == 0.05:  # Without this, won't match R exactly. R relies on 1.96, while I use SciPy
            zalpha = 1.96
        else:
            zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        # p-values are not implemented (doing my part to enforce CL over p-values)
        delta = np.where(self.df[self._missing_indicator] == 1, 1, 0)
        if self._continuous_outcome:
            # Calculating Average Treatment Effect
            Qstar = tmle_unit_unbound(Qstar, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar1 = tmle_unit_unbound(Qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar0 = tmle_unit_unbound(Qstar0, mini=self._continuous_min, maxi=self._continuous_max)

            self.average_treatment_effect = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            y_unbound = tmle_unit_unbound(self.df[self.outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = np.where(delta == 1,
                          HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.average_treatment_effect,
                          Qstar1 - Qstar0 - self.average_treatment_effect)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.average_treatment_effect_se = seIC
            self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * seIC,
                                                self.average_treatment_effect + zalpha * seIC]
            self.pval = (1- norm.cdf(np.abs(self.average_treatment_effect)/self.average_treatment_effect_se))*2
                # norm.cdf()

        else:
            # Calculating Risk Difference
            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          HAW * (self.df[self.outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference,
                          (Qstar1 - Qstar0) - self.risk_difference)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_difference_se = seIC
            self.risk_difference_ci = [self.risk_difference - zalpha * seIC,
                                       self.risk_difference + zalpha * seIC]

            # Calculating Risk Ratio
            self.risk_ratio = np.nanmean(Qstar1) / np.nanmean(Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          (1 / np.mean(Qstar1) * (H1W * (self.df[self.outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                           (1 / np.mean(Qstar0)) * (
                                       -1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0 - np.mean(Qstar0))),
                          (Qstar1 - np.mean(Qstar1)) + Qstar0 - np.mean(Qstar0))

            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_ratio_se = seIC
            self.risk_ratio_ci = [np.exp(np.log(self.risk_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.risk_ratio) + zalpha * seIC)]

            # Calculating Odds Ratio
            self.odds_ratio = (np.nanmean(Qstar1) / (1 - np.nanmean(Qstar1)
                                                     )) / (np.nanmean(Qstar0) / (1 - np.nanmean(Qstar0)))
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) *
                            (H1W * (self.df[self.outcome] - Qstar) + Qstar1)) -
                           (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) *
                            (-1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0))),

                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) * Qstar1 -
                            (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) * Qstar0))))
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.odds_ratio_se = seIC
            self.odds_ratio_ci = [np.exp(np.log(self.odds_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.odds_ratio) + zalpha * seIC)]

class tmle_baseline_test():
    def __init__(self,X,T,Y,n_bootstraps):
        X=X[:, ~(X == X[0, :]).all(0)]
        X = X[:,:25] #prevent numerical overflows
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] + ['Y']+['D']
        self.cov_string =''
        for i in range(self.d):
            self.cov_string+=f' + x_{i}'
        self.dfs = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        self.columns = columns
        self.n_bootstraps = n_bootstraps
        self.X,self.T,self.Y = X,T,Y
        self.tmle = TMLE_pval(self.dfs, exposure='D', outcome='Y')
        self.tmle.exposure_model(self.cov_string, print_results=False)
        self.tmle.outcome_model('D'+self.cov_string, print_results=False)
        self.tmle.fit()
        # tmle.summary()
        self.ref_stat = self.tmle.average_treatment_effect
    def permutation_test(self):
        # rd_results = []
        # for i in range(self.n_bootstraps):
        #     Y = np.random.permutation(self.Y)
        #     s = pd.DataFrame(np.concatenate([self.X,Y,self.T],axis=1),columns=self.columns)
        #     # s = self.dfs.sample(n=self.n, replace=True)
        #     tmle = TMLE(s, exposure='D', outcome='Y')
        #     tmle.exposure_model(self.cov_string, print_results=False)
        #     tmle.outcome_model('D' + self.cov_string, print_results=False)
        #     tmle.fit()
        #     rd_results.append(tmle.average_treatment_effect)
        # rd_results = np.array(rd_results)
        # print(rd_results)
        # pval=calculate_pval_symmetric(rd_results,self.ref_stat)
        pval = self.tmle.pval
        return pval,self.ref_stat


