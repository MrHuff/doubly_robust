import torch

from WMMD.Kernel import KGauss
from cvxopt import matrix, solvers
import math
import random
import numpy as np

from doubly_robust_method.kernels import Kernel
k_init=Kernel()

class WMMDTest:
    """
    Weighted MMD test where the null distribution is computed by permutation.
    """
    def __init__(self,X,T,Y, n_permute=250):
        """
        kernel:     an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between y                     samples 
        kernel_x: an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between x                     samples 
        n_permute:  number of times to do permutation
        """
        x_ls = k_init.get_median_ls(torch.from_numpy(X))
        y_ls = k_init.get_median_ls(torch.from_numpy(Y))

        self.kernel = KGauss(y_ls.item())
        self.kernel_x = KGauss(x_ls.item())
        self.n_permute = n_permute
        self.X=X
        self.Y=Y
        self.T=T
        self.n,self.d=X.shape

    def compute_weighted_mmd(self,Y,T,weights=None):

        k = self.kernel 

        if weights is None:

            weights = np.ones(self.n)

        Y0 = Y[[t==0 for t in T]]
        Y1 = Y[[t==1 for t in T]]
        K_00 = k.eval(Y0,Y0)
        K_01 = k.eval(Y0,Y1)
        K_11 = k.eval(Y1,Y1)
        K_00 = np.outer(weights,weights)*K_00
        K_01 = np.outer(weights,np.ones(Y1.shape[0]))*K_01
        n = K_00.shape[0]
        m = K_11.shape[0]

        mmd_squared = (np.sum(K_00) - np.trace(K_00)) / (n * (n - 1)) + (np.sum(K_11) - np.trace(K_11)) / (
                    m * (m - 1)) - 2 * np.sum(K_01) / (m * n)

        return mmd_squared

    def permutation_test(self):
        X=self.X
        T=self.T.squeeze()
        X0= X[T==0,:]
        X1= X[T==1,:]
        weights, _ = WMMDTest.kernel_mean_matching(X0, X1, self.kernel_x)
        self.test_stat= self.compute_weighted_mmd(self.Y ,T ,weights = weights)
        perm_stat=[]

        for i in range(self.n_permute):
            T= np.random.permutation(T).squeeze()
            X0= X[T==0,:]
            X1= X[T==1,:]
            weights, _ = WMMDTest.kernel_mean_matching(X0, X1, self.kernel_x)
            perm_stat.append(self.compute_weighted_mmd(self.Y ,T ,weights = weights))
        p_value = np.mean(self.test_stat > perm_stat)

        return p_value, self.test_stat

    @staticmethod
    def kernel_mean_matching(X1, X2, kx, B=10, eps=None):
        '''
        An implementation of Kernel Mean Matching, note that this implementation uses its own kernel parameter
        References:
        1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." 
        2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data."
        
        :param X1: two dimensional sample from population 1
        :param X2: two dimensional sample from population 2
        :param kern: kernel to be used, an instance of class Kernel in kernel_utils
        :param B: upperbound on the solution search space 
        :param eps: normalization error
        :return: weight coefficients for instances x1 such that the distribution of weighted x1 matches x2
        '''
        nx1 = X1.shape[0]
        nx2 = X2.shape[0]
        if eps == None:
            eps = B / math.sqrt(nx1)
        K = kx.eval(X1, X1)
        kappa = np.sum(kx.eval(X1, X2), axis=1) * float(nx1) / float(nx2)
        
        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, nx1)), -np.ones((1, nx1)), np.eye(nx1), -np.eye(nx1)])
        h = matrix(np.r_[nx1 * (1 + eps), nx1 * (eps - 1), B * np.ones((nx1,)), np.zeros((nx1,))])

        solvers.options['show_progress'] = False
        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])
        objective_value = sol['primal objective'] * 2 / (nx1**2) + np.sum(kx.eval(X2, X2)) / (nx2**2)
        
        return coef, objective_value
