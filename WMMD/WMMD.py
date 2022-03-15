import torch
from WMMD.Kernel import RBFKernel
from qpth.qp import QPFunction
import math
import random
import numpy as np
from torch.autograd import Variable


from doubly_robust_method.kernels import Kernel
k_init=Kernel()

class WMMDTest:
    """
    Weighted MMD test where the null distribution is computed by permutation.
    """
    def __init__(self,X,T,Y,device='cuda:0', n_permute=250):
        """
        kernel:     an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between y                     samples 
        kernel_x: an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between x                     samples 
        n_permute:  number of times to do permutation
        """
        x_ls = k_init.get_median_ls(torch.from_numpy(X))
        y_ls = k_init.get_median_ls(torch.from_numpy(Y))

        self.n_permute = n_permute
        self.B=10
        self.X=torch.from_numpy(X).float().to(device)
        self.Y = torch.from_numpy(Y).float().to(device)
        self.T = torch.from_numpy(T).float().to(device).squeeze()
        self.T_1 = (self.T==1).squeeze()
        self.nx1=int(torch.sum(self.T).item())
        self.n, self.d= self.X.shape
        self.eps= self.B/ math.sqrt(self.nx1)
        self.G = torch.from_numpy((np.r_[np.ones((1, self.nx1)), -np.ones((1, self.nx1)), np.eye(self.nx1), -np.eye(self.nx1)])).float().to(device)
        self.h = torch.from_numpy((np.r_[self.nx1 * (1 + self.eps), self.nx1 * (self.eps - 1), self.B * np.ones((self.nx1,)), np.zeros((self.nx1,))])).float().to(device)
        #self.G=(np.r_[np.ones((1, self.nx1)), -np.ones((1, self.nx1)), np.eye(self.nx1), -np.eye(self.nx1)])
        #self.h =((np.r_[self.nx1 * (1 + self.eps), self.nx1 * (self.eps - 1), self.B * np.ones((self.nx1,)), np.zeros((self.nx1,))]))
        self.h
        self.setup_Y_kernel(self.Y,'Y',device)
        self.setup_Y_kernel(self.X,'X',device)
        self.e=Variable(torch.Tensor())

    def setup_Y_kernel(self,Y,name,device):
        kernel = RBFKernel(Y).to(device)
        ls =RBFKernel.get_median_ls(Y,Y)
        kernel._set_lengthscale(ls)
        L = kernel.evaluate()
        setattr(self,name,L)
        setattr(self,'kernel_'+name,kernel)

    def compute_weighted_mmd(self,Y,T,weights=None):

        

        if weights is None:

            weights = np.ones(self.n)

        Y0 = Y[T==0,:]
        Y1 = Y[T==1,:]
        K_00 = self.kernel_Y(Y0,Y0)
        K_01 = self.kernel_Y(Y0,Y1)
        K_11 = self.kernel_Y(Y1,Y1)
        K_11 = (weights.T @ weights) * K_11
        K_01 = ( torch.ones(Y1.shape[0])@ weights.T)*K_01
        n = K_00.shape[0]
        m = K_11.shape[0]

        mmd_squared = (torch.sum(K_00) - torch.trace(K_00)) / (n * (n - 1)) + (torch.sum(K_11) - torch.trace(K_11)) / (
                    m * (m - 1)) - 2 * torch.sum(K_01) / (m * n)

        return mmd_squared.item()

    def permutation_test(self):
        X=self.X
        
        X0= X[~self.T_1,:]
        X1= X[self.T_1,:]
        weights= self.kernel_mean_matching(X1, X0)
        self.test_stat= self.compute_weighted_mmd(self.Y ,self.T ,weights = weights)
        perm_stat=[]

        for i in range(self.n_permute):
            T= self.T[torch.randperm(self.T.size()[0])]
            X0= X[T==0,:]
            X1= X[T==1,:]
            weights= self.kernel_mean_matching(X1, X0)
            perm_stat.append(self.compute_weighted_mmd(self.Y ,T ,weights = weights))
        perm_stat=np.array(perm_stat)
        p_value = np.mean(self.test_stat > perm_stat)

        return p_value, self.test_stat


    def kernel_mean_matching(self,X1, X2):
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
    
        K = self.kernel_X(X1, X1)
        kappa = torch.sum(self.kernel_X(X1, X2), axis=1) * float(self.nx1) / float(self.n-self.nx1)      
       
        coef = QPFunction(verbose=-1)(K, -kappa, self.G, self.h, self.e, self.e)
        
        return coef


