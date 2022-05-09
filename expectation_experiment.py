import itertools
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import torch
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
from doubly_robust_method.utils import *
from scipy.special import expit
from scipy.stats import bernoulli,uniform,expon,gamma
from baseline_cme.utils import gauss_rbf
from sklearn.metrics import pairwise_distances
import seaborn as sns
sns.set()
from matplotlib import rc
import tqdm
rc("text", usetex=False)

PI = np.pi

def linear(x):
    return x
nn_params = {
    'layers_x': [1],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': linear
}
nn_parms_2 = {
    'layers_x': [16, 8],
    'cat_size_list': [],
    'dropout': 0.0,
    'transformation': torch.tanh,
    'output_dim': 25,
}

def generate_observational_stuff_1(seed, ns, d, alpha_vec, alpha_0, beta_vec, b,noise_var=0.0,misspecified=False):
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    if misspecified:
        Prob_vec = expit(np.dot(alpha_vec, X.T)**2 + alpha_0)
    else:
        Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    noise= noise_var * np.random.randn(ns)
    Y = np.dot(beta_vec, X.T)  + b * T + noise
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis] , noise,None

def generate_interventional_stuff_1( X, beta_vec, b,T,noise,Z=None):
    Y = np.dot(beta_vec, X.T)  + b * T + noise
    YY = Y[:, np.newaxis]
    return YY

def generate_observational_stuff_2(seed, ns, d, alpha_vec, alpha_0, beta_vec, b,noise_var=0.0,misspecified=False):  # krik paper case 3
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    if misspecified:
        Prob_vec = expit(np.dot(alpha_vec, X.T)**2 + alpha_0)
    else:
        Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    noise= noise_var * np.random.randn(ns)
    T = bernoulli.rvs(Prob_vec)
    Z = bernoulli.rvs(0.5, size=len(T))
    Y = np.dot(beta_vec, X.T) +T*b*(2 * Z - 1) + noise
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis],noise,Z


def generate_interventional_stuff_2(X, beta_vec, b,T,noise,Z=None):  # krik paper case 3
    Y = np.dot(beta_vec, X.T) +b*T*(2 * Z - 1) + noise
    YY = Y[:, np.newaxis]
    return YY

def calc_r2(y_pred,y_true):
    pair = (y_true-y_pred)**2
    r2 = 1-pair.mean()/y_true.var()
    return r2.item(),pair.mean().item()

def calculate_and_simulate(seed,ns,b,sim_func,intervene_func,misspecified):
    training_params = {'bs': 100,
                       'patience': 10,
                       'device': 'cuda:0',
                       'permute_e': True,
                       'permutations': 250,
                       'oracle_weights': False,
                       'double_estimate_kme': False,
                       'epochs': 100,
                       'debug_mode': False,
                       'neural_net_parameters': nn_parms_2,
                       'approximate_inverse': False,
                       'neural_cme': False
                       }
    beta=1.0
    T, Y, X, W, noise,Z = sim_func(seed=seed, ns=ns, d=5,
                                                       alpha_vec=np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,
                                                       alpha_0=0.05,
                                                       beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * beta, b=b,misspecified=misspecified)
    T_test, Y_test, X_test, W_test, noise_test,Z = sim_func(seed=seed, ns=ns, d=5,
                                                                                alpha_vec=np.array(
                                                                                    [0.05, 0.04, 0.03, 0.02,
                                                                                     0.01]) * 20,
                                                                                alpha_0=0.05, beta_vec=np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5]) * beta, b=b,misspecified=misspecified)

    # X_0 = X_test[T_test.squeeze()==0,:]
    # X_1 = X_test[T_test.squeeze()==1,:]


    dr_c = testing_class(X=X, T=T, Y=Y, W=W, nn_params=nn_params, training_params=training_params)
    kme_0, kme_1 = dr_c.fit_class_and_embedding()
    mu_0, mu_1 = dr_c.compute_expectation(kme_0, kme_1, t_te=T_test, y_te=Y_test, x_te=X_test)  # #\mu_Y_{1}^DR(Y_test)

    x_ref_test= dr_c.tst_X
    tst_idx = dr_c.tst_idx

    Y_1_true = intervene_func(X=x_ref_test, beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * beta, b=b, T=1,
                                               noise=noise_test[tst_idx],Z=Z[tst_idx] if Z is not None else Z)
    Y_0_true = intervene_func(X=x_ref_test, beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * beta, b=b, T=0,
                                               noise=noise_test[tst_idx],Z=Z[tst_idx] if Z is not None else Z)

    L = dr_c.L_ker
    Y_1_true = torch.from_numpy(Y_1_true).float().cuda()
    Y_0_true = torch.from_numpy(Y_0_true).float().cuda()
    cuda_Y_tst = torch.from_numpy(Y_test).float().cuda()
    marginal_CFME_1 = L(cuda_Y_tst, Y_1_true).mean(1).squeeze()  # \mu_Y_{1}(Y_test)
    marginal_CFME_0 = L(cuda_Y_tst, Y_0_true).mean(1).squeeze() # \mu_Y_{0}(Y_test)
    r_2_1,mse_1 = calc_r2(mu_1.squeeze(), marginal_CFME_1)
    r_2_0,mse_0 = calc_r2(mu_0.squeeze(), marginal_CFME_0)

    c=baseline_test_class(X=X,T=T,Y=Y,W=W,nn_params=nn_params,training_params=training_params)
    c.fit_class_and_embedding()
    base_mu_0,base_mu_1=c.compute_expectation(t_te=T_test,y_te=Y_test,x_te=X_test)
    r_2_base_0,mse_base_0=calc_r2(base_mu_0.squeeze(),marginal_CFME_0)
    r_2_base_1,mse_base_1=calc_r2(base_mu_1.squeeze(),marginal_CFME_1)
    return r_2_0,r_2_1,r_2_base_0,r_2_base_1,mse_0,mse_1,mse_base_0,mse_base_1

#Tweakable parameters: b,distributional dataset, 5 seeds,n, to prove a point do 4 plots...
#x-axis = n
# y-axis error comparison ,with weights, not weights
# different plot for b= 0 non 0, distributional non-distributional
seeds = [1,2,3,4,5]
n_list = [100,250,500,1000,2000,5000]
ms = [False, True]
b_list = [0.0,0.5]
funcs = [['mean',generate_observational_stuff_1,generate_interventional_stuff_1],['distributional',generate_observational_stuff_2,generate_interventional_stuff_2]]

if __name__ == '__main__':
    fn_name='expectation_experiments_fixed_weights'
    if not os.path.exists(f'{fn_name}.csv'):
        params = list(itertools.product(seeds, n_list, ms, b_list, funcs))
        raw_data = []
        cols  = ['seed','n','mis','b','data type','DR 0 R^2','DR 1 R^2','mu 0 R^2','mu 1 R^2','DR 0 mse','DR 1 mse','mu 0 mse','mu 1 mse']
        for i,p in enumerate(tqdm.tqdm(params)):
            s,n,t,b,f = p
            name,obs_func,int_func = f
            r_2_0,r_2_1,r_2_base_0,r_2_base_1,mse_0,mse_1,mse_base_0,mse_base_1 = calculate_and_simulate(seed=s,ns=n,b=b,sim_func=obs_func,intervene_func=int_func,misspecified=t)
            raw_data.append([s,n,t,b,name,r_2_0, r_2_1, r_2_base_0, r_2_base_1,mse_0,mse_1,mse_base_0,mse_base_1])
        csv = pd.DataFrame(raw_data,columns=cols)
        csv.to_csv(f'{fn_name}.csv')
    else:
        if not os.path.exists(f'{fn_name}_plots'):
            os.makedirs(f'{fn_name}_plots')
        df = pd.read_csv(f'{fn_name}.csv')
        df[r'$\hat{\mu}_{Y(1)}$'] = df['mu 1 mse']
        df[r'$\hat{\mu}_{Y(0)}$'] = df['mu 0 mse']
        df[r'$\hat{\mu}_{Y(1)}^{DR}$'] = df['DR 1 mse']
        df[r'$\hat{\mu}_{Y(0)}^{DR}$'] = df['DR 0 mse']

        val_vars_list = [[r'$\hat{\mu}_{Y(1)}$',r'$\hat{\mu}_{Y(1)}^{DR}$'],[r'$\hat{\mu}_{Y(0)}$',r'$\hat{\mu}_{Y(0)}^{DR}$']]
        for i,val_vars in enumerate(val_vars_list):
            mean_mse = pd.melt(df, id_vars=['seed','n','mis','b','data type'], value_vars=val_vars)
            mean_mse['MSE'] = mean_mse['value']
            for mis in ms:
                for f in funcs:
                    f_s,_,_ =f
                    mask = (mean_mse["mis"] == mis) & (mean_mse['data type'] == f_s)
                    mask = mask.values
                    subset = mean_mse[mask]
                    g=sns.relplot(x="n", y="MSE", hue="variable", style="b", kind="line", data=subset)
                    g._legend.remove()
                    plt.savefig(f'{fn_name}_plots/fig_{mis}_{f_s}_{i}.png', bbox_inches ='tight')
                    plt.clf()



