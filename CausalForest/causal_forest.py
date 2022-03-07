import numpy as np
from econml.dml import CausalForestDML
from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric

class CausalForest_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,X_val,X_test,bootstrap=250):
        self.X_test=X_test
        self.n,self.d=X_test.shape
        self.est=CausalForestDML()
        self.est.fit(X=X_tr,T=T_tr,Y=Y_tr)
        self.ref_stat = self.est.ate(X_val)[0]
        self.bootstrap=bootstrap
        self.boostrap_size = int(round(self.X_test.shape[0]*0.1))

    def boostrap_data(self):
        ind_0=np.random.randint(0,self.X_test.shape[0],self.boostrap_size)
        return self.X_test[ind_0,:]

    def permutation_test(self):
        effect_list = []
        for i in range(self.bootstrap):
            x0 = self.boostrap_data()
            stat = self.est.ate(x0)
            effect_list.append(stat)
        effect_list = np.array(effect_list)
        pval = calculate_pval_symmetric(effect_list, self.ref_stat)

        return pval, self.ref_stat


