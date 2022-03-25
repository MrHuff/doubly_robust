

import numpy as np
import pandas as pd
from xbart import XBART


from vanilla_doublyrobust_baseline.vanilla_dr import calculate_pval_symmetric

class BART_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,bootstrap=250):
        x_tr_0, y_tr_0, x_tr_1, y_tr_1 = self.sep_dat(X_tr,Y_tr,T_tr)
        self.ref_stat = self.fit_BART(x_tr_0, y_tr_0, x_tr_1, y_tr_1 )
        self.bootstrap=bootstrap
        self.X_tr = X_tr
        self.T_tr = T_tr
        self.Y_tr = Y_tr

    def fit_BART(self,x_tr_0,y_tr_0,x_tr_1, y_tr_1):
        est_0 = XBART(num_trees=100, num_sweeps=10, burnin=1)
        est_1 = XBART(num_trees=100, num_sweeps=10, burnin=1)
        est_0.fit(x_tr_0, y_tr_0.squeeze())
        est_1.fit(x_tr_1, y_tr_1.squeeze())
        return self.calc_effect(x_tr_0,x_tr_1,est_0,est_1)


    def calc_effect(self,x0,x1,est_0,est_1):
        ref_mat_0_0 = est_0.predict(x0)
        ref_mat_0_1 = est_0.predict(x1)
        ref_mat_1_0 = est_1.predict(x0)
        ref_mat_1_1 = est_1.predict(x1)
        pred_0=np.concatenate([ref_mat_0_0,ref_mat_0_1],axis=0)
        pred_1=np.concatenate([ref_mat_1_0,ref_mat_1_1],axis=0)
        #concat vectors then take mean
        return (pred_1-pred_0).mean()

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
    #
    # def boostrap_data(self):
    #     ind_0=np.random.randint(0,self.x_test_0.shape[0],self.boostrap_size)
    #     ind_1=np.random.randint(0,self.x_test_1.shape[0],self.boostrap_size)
    #     return self.x_test_0[ind_0,:],self.x_test_1[ind_1,:]

    def permutation_test(self):
        effect_list=[]
        for i in range(self.bootstrap):
            Y = np.random.permutation(self.Y_tr)

            x_tr_0, y_tr_0, x_tr_1, y_tr_1 = self.sep_dat(self.X_tr, Y, self.T_tr)
            stat = self.fit_BART(x_tr_0, y_tr_0, x_tr_1, y_tr_1)
            effect_list.append(stat)

        effect_list=np.array(effect_list)
        pval=calculate_pval_symmetric(effect_list,self.ref_stat)

        return pval,self.ref_stat


