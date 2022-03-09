import numpy as np
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd

class doubleML_baseline_test():
    def __init__(self,X,Y,T):
        n,d=X.shape
        ml_g = RandomForestRegressor(n_estimators=100, max_features=d, max_depth=5, min_samples_leaf=2)
        ml_m = RandomForestClassifier(n_estimators=100, max_features=d, max_depth=5, min_samples_leaf=2)
        columns=[f'x_{i}' for i in range(d)] + ['Y']+['D']
        data = pd.DataFrame(np.concatenate([X,Y,T],axis=1),columns=columns)
        obj_dml_data = dml.DoubleMLData(data, 'Y', 'D')
        self.dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)

    def permutation_test(self):
        self.dml_irm_obj.fit()
        pval= self.dml_irm_obj.summary['P>|t|'].item()
        stat= self.dml_irm_obj.summary['t'].item()
        return pval,stat


