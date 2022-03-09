import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd

def calculate_pval_symmetric(bootstrapped_list, test_statistic):
    pval_right = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
    pval_left = 1 - pval_right
    pval = 2 * min([pval_left.item(), pval_right.item()])
    return pval

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

class vanilla_dr_baseline_test():
    def __init__(self,X,T,Y,n_bootstraps):
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] +['D']+['Y']
        # self.cov_string =''
        # for i in range(self.d):
        #     self.cov_string+=f' + x_{i}'
        self.X,self.T,self.Y = X,T,Y
        self.columns = columns
        self.dfs = pd.DataFrame(np.concatenate([X,T,Y],axis=1),columns=columns)
        self.n_bootstraps = n_bootstraps
        self.x_col = [f'x_{i}' for i in range(self.d)]
        self.ref_stat = doubly_robust(self.dfs,X=self.x_col,T='D',Y='Y')
    #TODO, when permuting just go back to what we did
    def permutation_test(self):
        rd_results = []
        for i in range(self.n_bootstraps):
            Y = np.random.permutation(self.Y)
            s = pd.DataFrame(np.concatenate([self.X,self.T,Y],axis=1),columns=self.columns)
            stat = doubly_robust(s,self.x_col,'D','Y')
            rd_results.append(stat)
        rd_results = np.array(rd_results)
        pval=calculate_pval_symmetric(rd_results,self.ref_stat)
        return pval,self.ref_stat
