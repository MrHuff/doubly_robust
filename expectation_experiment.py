from doubly_robust_method.utils import *
from scipy.special import expit
from scipy.stats import bernoulli,uniform,expon,gamma
from baseline_cme.utils import gauss_rbf
from sklearn.metrics import pairwise_distances
PI = np.pi
''
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



def generate_observational_stuff_1(seed, ns, d, alpha_vec, alpha_0, beta_vec, b):
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Y = np.dot(beta_vec, X.T)  + b * T
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]

def generate_interventional_stuff_1( X, beta_vec, b,T):
    Y = np.dot(beta_vec, X.T)  + b * T
    YY = Y[:, np.newaxis]
    return YY

def generate_observational_stuff_2(seed, ns, d, alpha_vec, alpha_0, beta_vec, b):  # krik paper case 3
    np.random.seed(seed)
    X = np.random.randn(ns, d)
    Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
    T = bernoulli.rvs(Prob_vec)
    Z = bernoulli.rvs(0.5, size=len(T))
    Y = np.dot(beta_vec, X.T) +b*T*(2 * Z - 1)
    YY = Y[:, np.newaxis]
    return T[:, np.newaxis], YY, X, Prob_vec.squeeze()[:, np.newaxis]


def generate_interventional_stuff_2(X,Z, beta_vec, b,T):  # krik paper case 3
    Y = np.dot(beta_vec, X.T) +b*T*(2 * Z - 1)
    YY = Y[:, np.newaxis]
    return YY

ref_dict={'seed': 0,
         'ns': 0,
         'd': 0,
         'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,  # Treatment assignment
         # the thing just blows up regardless of what you do?!
         # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
         'alpha_0': 0.05,  # 0.05,
         'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
         'noise_var': 0.1,
         'b': 0
         }
if __name__ == '__main__':
    seed=1
    b=0.1
    ns=1000
    neural_cme =False
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
                       'neural_cme': neural_cme
                       }

    T,Y,X,W=generate_observational_stuff_1(seed=seed,ns=2*ns,d=5,alpha_vec= np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,
                                 alpha_0=0.05,beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,b=b)
    T_test, Y_test, X_test, W_test = generate_observational_stuff_1(seed=seed, ns=ns, d=5,
                                                alpha_vec=np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 20,
                                                alpha_0=0.05, beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05, b=b)

    # X_0 = X_test[T_test.squeeze()==0,:]
    # X_1 = X_test[T_test.squeeze()==1,:]
    Y_1_true=generate_interventional_stuff_1(X=X_test,beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,b=b,T=1)
    # Y_0_true=generate_interventional_stuff_1(X=X_1,beta_vec=np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,b=b,T=0)

    dr_c=testing_class(X=X,T=T,Y=Y,W=W,nn_params=nn_params,training_params=training_params)
    kme_0,kme_1=dr_c.fit_class_and_embedding()
    mu_0,mu_1=dr_c.compute_expectation(kme_0,kme_1,t_te=T_test,y_te=Y_test,x_te=X_test)

    c=baseline_test_class(X=X,T=T,Y=Y,W=W,nn_params=nn_params,training_params=training_params)
    c.fit_class_and_embedding()
    base_mu_0,base_mu_1=c.compute_expectation(t_te=T_test,y_te=Y_test,x_te=X_test)
    # print(base_mu_1)
    print(mu_1)
    print(Y_1_true)


    # testing_class
    # baseline_test_class

