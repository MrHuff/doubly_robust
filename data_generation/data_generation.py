import numpy as np
from scipy.special import expit
from scipy.stats import bernoulli

# generate data from the marginal distributions P(X_0) and P(X_1)
# ns = 50
# d  = 5
# noise_var = 0.1
# # generate Y_0 and Y_1 from the conditional models
# beta_vec  = np.array([0.1,0.2,0.3,0.4,0.5])
# alpha_vec = np.array([0.05,0.04,0.03,0.02,0.01])
# alpha_0   = 0.05
#
# significance_level = 0.01
#
# b=0

def case_1(seed,ns,d,alpha_vec,alpha_0,beta_vec,noise_var,b): #krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T) + noise_var*np.random.randn(ns)+b*T
    YY= Y[:, np.newaxis]
    return T[:,np.newaxis],YY,X

def case_1_ref(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Y0 = np.dot(beta_vec, X[T == 0, :].T) + noise_var * np.random.randn(X[T == 0, :].shape[0])
    Y1 = np.dot(beta_vec, X[T == 1, :].T) + b + noise_var * np.random.randn(X[T == 1, :].shape[0])
    YY0 = Y0[:, np.newaxis]
    YY1 = Y1[:, np.newaxis]
    return YY0,YY1

def case_2(seed,ns,d,alpha_vec,alpha_0,beta_vec,noise_var,b): #krik paper case 3
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z = bernoulli.rvs(0.5, size=len(T))
    Y = np.dot(beta_vec, X.T) + (2 * Z - 1) +noise_var*np.random.randn(ns)
    YY= Y[:, np.newaxis]
    return T,YY,X


def case_3(): #breaking cme's
    pass


