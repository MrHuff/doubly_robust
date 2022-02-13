import numpy as np
from sklearn.model_selection import train_test_split
from doubly_robust_method.propensity_classifier import *
from doubly_robust_method.kme import *
from doubly_robust_method.test_statistic import *
import os
from scipy.stats import kstest
import pickle
from baseline_cme.baseline_test_object import *

class testing_class():
    def __init__(self,X,T,Y,W,nn_params,training_params,cat_cols=[]): #assuming data comes in as numpy
        #split data
        self.cat_cols=cat_cols
        self.unique_cat_cols=[]
        if cat_cols:
            self.cont_cols = list(set([i for i in range(X.shape[1])])-set(cat_cols))
            self.X_cont,self.X_cat,self.unique_cat_cols=categorical_transformer(X,self.cat_cols,cont_cols=self.cont_cols)
        self.nn_params = nn_params
        self.nn_params['cat_size_list'] = self.unique_cat_cols
        self.nn_params['d_in_x'] = self.X_cont.shape[1] if cat_cols else X.shape[1]
        indices = np.arange(X.shape[0])
        tr_ind, tst_ind, tmp_T, self.tst_T = train_test_split(indices,T, test_size = 0.5,stratify=T)
        tr_ind_2, val_ind, self.tr_T, self.val_T = train_test_split(tr_ind,tmp_T, test_size = 0.1,stratify=tmp_T)
        self.training_params = training_params
        self.split_into_cont_cat(X,Y,W,tr_ind_2,val_ind,tst_ind)

    def split_into_cont_cat(self,X,Y,W,tr_ind_2,val_ind,tst_ind):
        kw = ['tr','val','tst']
        indices = [tr_ind_2,val_ind,tst_ind]
        for w,idx in zip(kw,indices):
            setattr(self,f'{w}_X',X[idx])
            setattr(self,f'{w}_Y',Y[idx])
            setattr(self,f'{w}_W',W[idx])
            if self.cat_cols:
                setattr(self, f'{w}_X_cont', self.X_cont[idx])
                setattr(self, f'{w}_X_cat', self.X_cat[idx])
            else:
                setattr(self, f'{w}_X_cat',[])
                setattr(self, f'{w}_X_cont', X[idx])


    def fit_y_cond_x(self,tr_X,tr_Y,tr_T,val_X,val_Y,val_T):
        kme_1=kme_model(tr_X,tr_Y,tr_T,val_X,val_Y,val_T,treatment_const=1,device=self.training_params['device'])
        kme_1.fit()
        print('cme 1 val error: ', kme_1.best)
        print('cme 1 lamb: ', kme_1.best_lamb)
        print('cme 1 ls: ', kme_1.kernel.ls.item())
        kme_0=kme_model(tr_X,tr_Y,tr_T,val_X,val_Y,val_T,treatment_const=0,device=self.training_params['device'])
        kme_0.fit()
        print('cme 0 val error: ', kme_0.best)
        print('cme 0 lamb: ', kme_0.best_lamb)
        print('cme 0 ls: ', kme_0.kernel.ls.item())
        return kme_0,kme_1


    def run_test(self,seed):
        #train classifier
        self.classifier = propensity_estimator(self.tr_X_cont,self.tr_T,self.val_X_cont,
                                               self.val_T,nn_params=self.nn_params,
                                               bs=self.training_params['bs'],X_cat_val=self.val_X_cat,X_cat_tr=self.tr_X_cat,epochs=self.training_params['epochs'])

        if self.training_params['oracle_weights']:
            self.e = torch.from_numpy(self.tst_W)
        else:
            self.classifier.fit(self.training_params['patience'])
            print('classifier val auc: ', self.classifier.best)
            self.e = self.classifier.predict(self.tst_X,self.tst_T,self.tst_X_cat)

        self.kme_0,self.kme_1 = self.fit_y_cond_x(self.tr_X,self.tr_Y,self.tr_T,self.val_X,self.val_Y,self.val_T)

        if self.training_params['double_estimate_kme']:

            perm_tr=torch.randperm(self.tr_X.shape[0])
            perm_val=torch.randperm(self.val_X.shape[0])
            self.kme_0_indep,self.kme_1_indep = self.fit_y_cond_x(self.tr_X,self.tr_Y[perm_tr],self.tr_T,self.val_X,self.val_Y[perm_val],self.val_T)
        else:
            self.kme_0_indep, self.kme_1_indep=self.kme_0,self.kme_1
        #run test
        self.test = counterfactual_me_test(X=self.tst_X,Y=self.tst_Y,e=self.e,T=self.tst_T,kme_1=self.kme_1,kme_0=self.kme_0,
                                           kme_0_indep=self.kme_0_indep,
                                           kme_1_indep=self.kme_1_indep,
                                           permute_e=self.training_params['permute_e'],
                                           permutations=self.training_params['permutations'],
                                           device=self.training_params['device'])

        perm_stats,self.tst_stat = self.test.permutation_test()
        self.perm_stats = np.array(perm_stats)
        self.pval = self.calculate_pval_symmetric(self.perm_stats,self.tst_stat )
        output = [seed,self.pval,self.tst_stat]
        return output+perm_stats

    @staticmethod
    def calculate_pval_right_tail(bootstrapped_list, test_statistic):
        pval = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        return pval.item()
    @staticmethod
    def calculate_pval_left_tail(bootstrapped_list, test_statistic):
        pval = 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        return pval.item()
    @staticmethod
    def calculate_pval_symmetric(bootstrapped_list, test_statistic):
        pval_right = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
        pval_left = 1 - pval_right
        pval = 2 * min([pval_left.item(), pval_right.item()])
        return pval

class baseline_test_class(testing_class):
    def __init__(self,X,T,Y,W,nn_params,training_params,cat_cols=[]):
        super(baseline_test_class, self).__init__(X,T,Y,W,nn_params,training_params,cat_cols=cat_cols)

    def run_test(self,seed):
        #train classifier

        if self.training_params['oracle_weights']:
            self.e = torch.from_numpy(self.tst_W)
        else:
            self.classifier = propensity_estimator(self.tr_X_cont, self.tr_T, self.val_X_cont,
                                                   self.val_T, nn_params=self.nn_params,
                                                   bs=self.training_params['bs'], X_cat_val=self.val_X_cat,
                                                   X_cat_tr=self.tr_X_cat, epochs=self.training_params['epochs'])

            self.classifier.fit(self.training_params['patience'])
            print('classifier val auc: ', self.classifier.best)
            self.e = self.classifier.predict(self.tst_X,self.tst_T,self.tst_X_cat)
        #train KME
        self.test=baseline_test(self.tst_Y,e=self.e.cpu(),T=self.tst_T,permutations=self.training_params['permutations'])
        perm_stats,self.tst_stat = self.test.permutation_test()
        self.perm_stats = perm_stats
        self.pval = self.calculate_pval_symmetric(self.perm_stats,self.tst_stat )
        output = [seed,self.pval,self.tst_stat]
        return output+perm_stats.tolist()

    #TODO: calc size,power and KS-test

class experiment_object:
    def __init__(self,experiment_save_path,data_dir_load,num_exp,nn_params,training_params,cat_cols,test_type='baseline',debug_mode=False):
        self.experiment_save_path= experiment_save_path
        if not os.path.exists(experiment_save_path):
            os.makedirs(experiment_save_path)
        self.data_file_list=os.listdir(data_dir_load)
        self.debug_mode=debug_mode
        self.num_exp = num_exp
        self.data_dir_load = data_dir_load
        self.nn_params = nn_params
        self.training_params = training_params
        self.cat_cols = cat_cols
        self.test_type = test_type
    @staticmethod
    def get_level(level, p_values):
        total_pvals = len(p_values)
        power = sum(p_values <= level) / total_pvals
        return power

    def calc_summary_stats(self,pvals):
        output=[]
        ks_stat, ks_pval = kstest(pvals, 'uniform')
        levels = [0.01, 0.05, 0.1]
        for l in levels:
            output.append(self.get_level(l,pvals))
        output.append(ks_pval)
        output.append(ks_stat)
        return output

    def run_experiments(self):
        summary_job_cols = ['pow_001','pow_005','pow_010','KS-pval','KS-stat']
        columns = ['seed','pval','tst_stat']+[f'perm_{i}' for i in range(self.training_params['permutations'])]
        data_col = []
        counter_error_lim = 10
        c = 0
        for i,files in enumerate(self.data_file_list):
            with open(self.data_dir_load+'/'+files, 'rb') as handle:
                data_dict = pickle.load(handle)
            seed=data_dict['seed']
            T=data_dict['T']
            Y=data_dict['Y']
            X=data_dict['X']
            W=data_dict['W']
            if self.test_type=='doubly_robust':
                tst = testing_class(X=X,Y=Y,T=T,W=W,nn_params=self.nn_params,training_params=self.training_params,cat_cols=self.cat_cols)
            elif self.test_type=='baseline':
                tst = baseline_test_class(X=X,Y=Y,T=T,W=W,nn_params=self.nn_params,training_params=self.training_params,cat_cols=self.cat_cols)

            if self.debug_mode:
                out = tst.run_test(seed)
                data_col.append(out)
            else:
                try:
                    out = tst.run_test(seed)
                    data_col.append(out)
                except Exception as e:
                    print(e)
                    c+=1
                if c>counter_error_lim:
                    raise Exception('Dude something is seriously wrong with your data or the method please debug')
        big_df = pd.DataFrame(data_col,columns=columns)
        pvals = big_df['pval'].values
        output = self.calc_summary_stats(pvals)
        final_res = pd.DataFrame([output],columns=summary_job_cols)
        big_df.to_csv(f'{self.experiment_save_path}/big_df.csv')
        final_res.to_csv(f'{self.experiment_save_path}/final_res.csv')











