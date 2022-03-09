import numpy as np
from econml.dml import CausalForestDML
from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric

class CausalForest_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,bootstrap=250):
        self.n,self.d=X_tr.shape
        self.est=CausalForestDML()
        self.est.fit(X=X_tr,T=T_tr,Y=Y_tr)
        self.ref_stat = self.est.ate(X_tr)[0]
        self.bootstrap=bootstrap
        self.X_tr = X_tr
        self.T_tr = T_tr
        self.Y_tr = Y_tr

    # def boostrap_data(self):
    #     ind_0=np.random.randint(0,self.X_test.shape[0],self.boostrap_size)
    #     return self.X_test[ind_0,:]

    def permutation_test(self):
        effect_list = []
        for i in range(self.bootstrap):
            Y = np.random.permutation(self.Y_tr)
            est= CausalForestDML()
            est.fit(X=self.X_tr,T=self.T_tr,Y=Y)
            stat = self.est.ate(self.X_tr)
            effect_list.append(stat)
        effect_list = np.array(effect_list)
        pval = calculate_pval_symmetric(effect_list, self.ref_stat)

        return pval, self.ref_stat


