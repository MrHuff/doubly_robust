
import numpy as np
from sklearn.model_selection import train_test_split
from propensity_classifier import *
from kme import *
from test_statistic import *


#TODO: fix cat_cols for the pipeline, fix prediction of weights
class testing_class():
    def __init__(self,X,T,Y,nn_params,training_params,cat_cols=[]): #assuming data comes in as numpy
        #split data
        indices = np.arange(X.shape[0])
        tr_ind, tst_ind, tmp_T, self.tst_T = train_test_split(indices,T, test_size = 0.5,stratify=T)
        tmp_X=X[tr_ind]
        tmp_Y=Y[tr_ind]
        self.tst_X=X[tst_ind]
        self.tst_Y=Y[tst_ind]
        tr_indices =np.arange(tmp_X.shape[0])
        tr_ind, val_ind, self.tr_T, self.val_T = train_test_split(tr_indices,tmp_T, test_size = 0.1,stratify=tmp_T)
        self.tr_X=tmp_X[tr_ind]
        self.tr_Y=tmp_Y[tr_ind]
        self.val_X = tmp_X[val_ind]
        self.val_Y = tmp_Y[val_ind]
        self.training_params = training_params
        self.nn_params = nn_params
        self.cat_cols=cat_cols
        #train classifier


    def run_test(self):
        self.classifier = propensity_estimator(self.tr_X,self.tr_T,self.val_X,self.val_T,nn_params=self.nn_params,bs=self.training_params['bs'])
        self.classifier.fit(self.training_params['patience'])
        self.e = self.classifier.predict(self.tst_X,self.tst_cat_X)
        #train KME
        self.kme=kme_model(self.tr_X,self.tr_Y,self.tr_T,self.val_X,self.val_Y,self.val_T,self.training_params['device'])
        #run test
        self.test = counterfactual_me_test(X=self.tst_X,Y=self.tst_Y,e=self.e,T=self.tst_T,kme_object=self.kme,
                                           permute_e=self.training_params['permute_e'],
                                           permutations=self.training_params['permutations'],
                                           device=self.training_params['device'])

        perm_stats,tst_stat = self.test.permutation_test()

        return perm_stats,tst_stat


