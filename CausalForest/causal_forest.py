import numpy as np
from econml.dml import CausalForestDML
from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric

class CausalForest_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,bootstrap=250):
        self.n,self.d=X_tr.shape
        self.est=CausalForestDML(discrete_treatment=True,drate=True)
        self.est.fit(X=X_tr,T=T_tr,Y=Y_tr,)
        self.ref_stat = self.est.ate(X_tr)[0]
        self.bootstrap=bootstrap
        self.X_tr = X_tr
        self.T_tr = T_tr
        self.Y_tr = Y_tr

    def permutation_test(self):
        inf_object = self.est.ate__inference()
        return inf_object.pvalue()[0][0], self.ref_stat


