import numpy as np
from scipy.special import expit
from scipy.stats import bernoulli,uniform,expon,gamma
from baseline_cme.utils import gauss_rbf
from sklearn.metrics import pairwise_distances
PI = np.pi
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

# def U_func(fam_y,X,a,b):
#     if fam_y == 1:  # U-shape
#         if torch.is_tensor(b):
#             p = Normal(loc=a + (X ** 2) @ b, scale=1)  # Consider square matrix valued b.
#         else:
#             p = Normal(loc=a + (X ** 2) * b, scale=1)  # Consider square matrix valued b.
#     if fam_y == 2:  # V-shape
#         if torch.is_tensor(b):
#             p = Normal(loc=a + torch.abs(X) @ b, scale=1)  # Consider square matrix valued b.
#         else:
#             p = Normal(loc=a + torch.abs(X) * b, scale=1)  # Consider square matrix valued b.
#     if fam_y == 3:  # V-shape
#         if torch.is_tensor(b):
#             p = Normal(loc=a + X @ b, scale=1)  # Consider square matrix valued b.
#         else:
#             p = Normal(loc=a + X * b, scale=1)  # Consider square matrix valued b.
#     if fam_y == 4:  # Sin shape
#         trans_x = torch.exp(-(X ** 2) / 10) * torch.cos(X * PI)
#         if torch.is_tensor(b):
#             var = 1.0 if b[0]==0 else b[0]/5
#             p = Normal(loc=a + trans_x @ b, scale=var)  # Consider square matrix valued b.
#         else:
#             var = 1.0 if b==0 else b/5
#             p = Normal(loc=a + trans_x * b, scale=var)  # Consider square matrix valued b.
#
#     return p

def case_0(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = np.array([0.5]*ns)
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T) + noise_var * np.random.randn(ns) + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]



#Remove T dependency for expetation calculation

def case_1(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    # Prob_vec = np.exp(-np.dot(alpha_vec, X.T)**2)
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T) + noise_var * np.random.randn(ns) + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]

def case_1_robin(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)

    _censoring_idx = np.random.randint(0,ns,round(ns*0.1))

    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    Prob_vec[_censoring_idx]=0.0
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T) + noise_var * np.random.randn(ns) + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def case_1_xy_banana(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    # Prob_vec = np.exp(-np.dot(alpha_vec, X.T)**2)
    T = bernoulli.rvs(Prob_vec)

    X_banana = X**2
    Y = np.dot(beta_vec, X_banana.T) + noise_var * np.random.randn(ns) + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def case_1_xy_sin(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    # Prob_vec = np.exp(-np.dot(alpha_vec, X.T)**2)
    T = bernoulli.rvs(Prob_vec)

    X_SINC = np.exp(-(X ** 2) / 10) * np.cos(X * PI)

    Y = np.dot(beta_vec, X_SINC.T) + noise_var * np.random.randn(ns) + b * T
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


# alpha_vec_2 = np.array([1.05,1.04,1.03,1.02,1.01])/50.
def case_break_weights(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, (X ** 2).T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    y_cond_x = np.dot(beta_vec,X.T)
    Y = y_cond_x  + noise_var * np.random.randn(ns) + b * T #make CME get the job done on Y|X
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def case_1b(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 1,2
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

def case_distributional(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # krik paper case 3
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z = bernoulli.rvs(0.5, size=len(T))
    Y = np.dot(beta_vec, X.T) +b*T*(2 * Z - 1) + noise_var * np.random.randn(ns)
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]

def case_distributional_2(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z = uniform.rvs(loc=0,scale=2,size=len(T))
    Y = np.dot(beta_vec, X.T) +b*T*(2*Z-2) + noise_var * np.random.randn(ns)
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]

def case_distributional_3(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z_1 = gamma.rvs(a=1,loc=2,scale=2,size=len(T))*0.1
    mean, var, skew, kurt = gamma.stats(a=1,loc=2,scale=2, moments='mvsk')
    print(mean)
    # Z_1 = expon.rvs(loc=0.5,scale=0.5,size=len(T))*0.1
    Z_2 = bernoulli.rvs(0.5, size=len(T))

    Y = np.dot(beta_vec, X.T) +b*T*((2*Z_2-1)*0.1 + (-1)**(1-Z_2)*Z_1) + noise_var * np.random.randn(ns)
    YY = Y[:, np.newaxis]
    print(YY.mean())
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]



# def case_distributional_bdhsic(seed, ns, d, alpha_vec, alpha_0, beta_vec, noise_var, b):  # breaking cme's i.e. bd-HSIC case
#     np.random.seed(seed)
#     X = np.random.randn(ns, d)
#     sig_X = expit(X)
#     sig_X_neg = expit(-X)
#     T = bernoulli.rvs(sig_X)
#     Y =  np.random.normal((2 * T - 1) * np.abs(X) * b, 0.05 ** 0.5,ns)
#     Prob_vec = np.where(T == 1, 1 / (2 * sig_X), 1 / (2 * sig_X_neg))
#     YY = Y[:, np.newaxis]
#     return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]




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
    # mis = torch.where(x_samples == 1, 1 / (2 * sig_z), 1 / (2 * sig_z_neg))
    # mis = mis.prod(dim=1)
    # return x_samples,y_samples,z_samples,mis
