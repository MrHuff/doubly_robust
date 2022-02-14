import numpy as np
from scipy.special import expit
from scipy.stats import bernoulli
from baseline_cme.utils import gauss_rbf
from sklearn.metrics import pairwise_distances

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

def case_1(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T) + noise_var * np.random.randn(ns) + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def case_1_ref(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Y0 = np.dot(beta_vec, X[T == 0, :].T) + noise_var * np.random.randn(X[T == 0, :].shape[0])
    Y1 = np.dot(beta_vec, X[T == 1, :].T) + b + noise_var * np.random.randn(X[T == 1, :].shape[0])
    YY0 = Y0[:, np.newaxis]
    YY1 = Y1[:, np.newaxis]
    return YY0, YY1, Prob_vec

def case_1a(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    sigma2 = np.median(pairwise_distances(X, X, metric='euclidean')) ** 2
    x_ker = gauss_rbf(X[:d,:],X,sigma2)
    y_cond_x = np.dot(beta_vec,x_ker)
    Y = y_cond_x  + noise_var * np.random.randn(ns) + b * T #make CME get the job done on Y|X
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]

def case_2(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 3
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z = bernoulli.rvs(0.5, size=len(T))
    Y = np.dot(beta_vec, X.T) + (2 * Z - 1) + noise_var * np.random.randn(ns)
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def case_3(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # breaking cme's i.e. bd-HSIC case
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    sig_X = expit(X)
    sig_X_neg = expit(-X)
    T = bernoulli.rvs(sig_X)
    Y =  np.rando.normal((2 * T - 1) * np.abs(X) * b, 0.05 ** 0.5,ns)
    Prob_vec = np.where(T == 1, 1 / (2 * sig_X), 1 / (2 * sig_X_neg))
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


    # torch.random.manual_seed(seed)
    # z_dist = Normal(0, 1)
    # z_samples = z_dist.sample((n, d))
    # sig_z = torch.sigmoid(z_samples)
    # sig_z_neg = torch.sigmoid(-z_samples)
    # x_dist = Bernoulli(probs=sig_z)
    # x_samples = x_dist.sample(())
    # if null_case:
    #     y_dist = Normal(z_samples * alp, 0.05 ** 0.5)
    # else:
    #     y_dist = Normal((2 * x_samples - 1) * z_samples.abs() * alp, 0.05 ** 0.5)
    #
    # y_samples = y_dist.sample(())
    # if seed == 1:
    #     plt_y_marg = y_samples.numpy()
    #     plt.hist(plt_y_marg, bins=100)
    #     plt.savefig(new_dirname + '/y_marg.png')
    #     plt.clf()
    #
    # w = torch.where(x_samples == 1, 1 / (2 * sig_z), 1 / (2 * sig_z_neg))
    # w = w.prod(dim=1)
    # return x_samples,y_samples,z_samples,w
