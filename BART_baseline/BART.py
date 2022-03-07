

import numpy as np
import pandas as pd
from xbart import XBART

from doubly_robust_method.utils import testing_class


class BART_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,X_val,T_val,X_test,T_test,bootstrap=250):
        x_tr_0, y_tr_0, x_tr_1, y_tr_1 = self.sep_dat(X_tr,Y_tr,T_tr)
        self.est_0=XBART(num_trees=100, num_sweeps=40, burnin=15)
        self.est_0.fit(x_tr_0,y_tr_0.squeeze())
        self.est_1=XBART(num_trees=100, num_sweeps=40, burnin=15)
        self.est_1.fit(x_tr_1,y_tr_1.squeeze())
        x_val_0, x_val_1=self.sep_val(X_val,T_val)
        self.x_test_0, self.x_test_1=self.sep_val(X_test,T_test)
        self.ref_stat = self.calc_effect(x_val_0,x_val_1)
        self.boostrap_size = int(round(min(self.x_test_0.shape[0],self.x_test_1.shape[0])*0.1))
        self.bootstrap=bootstrap

    def calc_effect(self,x0,x1):
        ref_mat_0 = self.est_0.predict(x0)
        ref_mat_1 = self.est_1.predict(x1)
        yhat_0= ref_mat_0.mean()
        yhat_1= ref_mat_1.mean()
        return yhat_1-yhat_0

    def sep_dat(self,X,Y,T):
        mask_0 = (T==0).squeeze()
        x_tr_0,y_tr_0= X[mask_0,:],Y[mask_0]
        x_tr_1,y_tr_1= X[~mask_0,:],Y[~mask_0]
        return x_tr_0,y_tr_0,x_tr_1,y_tr_1

    def sep_val(self,X,T):
        mask_0 = (T==0).squeeze()
        x_tr_0 = X[mask_0,:]
        x_tr_1= X[~mask_0,:]
        return x_tr_0,x_tr_1

    def boostrap_data(self):
        ind_0=np.random.randint(0,self.x_test_0.shape[0],self.boostrap_size)
        ind_1=np.random.randint(0,self.x_test_1.shape[0],self.boostrap_size)
        return self.x_test_0[ind_0,:],self.x_test_1[ind_1,:]

    def permutation_test(self):
        effect_list=[]
        for i in range(self.bootstrap):
            x0,x1=self.boostrap_data()
            stat=self.calc_effect(x0,x1)
            effect_list.append(stat)

        effect_list=np.array(effect_list)
        pval=testing_class.calculate_pval_symmetric(effect_list,self.ref_stat)

        return pval,self.ref_stat


