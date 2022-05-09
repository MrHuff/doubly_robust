import numpy as np
import torch
from doubly_robust_method.kernels import *
from baseline_cme.kernel_two_sample_test_nonuniform import kernel_two_sample_test_nonuniform
from baseline_cme.kernel_two_sample_test_nonuniform_gpu import kernel_two_sample_test_nonuniform_gpu
from baseline_cme.kernel_two_sample_test_nonuniform_gpu_correct import kernel_two_sample_test_nonuniform_gpu_correct
from sklearn.metrics import pairwise_distances
general_ker_obj =Kernel()

class baseline_test():
    def __init__(self,Y,e,T,permutations = 250):
        self.YY0=Y[T==0][:,np.newaxis]
        self.YY1=Y[T==1][:,np.newaxis]
        self.sigma2= np.median(pairwise_distances(self.YY0, self.YY1, metric='euclidean')) ** 2
        e_0 = e[T==0].numpy()
        e_1 = e[T==1].numpy()
        self.e_input = np.concatenate([e_0,e_1],axis=0)
        self.perms = permutations


    def permutation_test(self):
        mmd2u_rbf, mmd2u_null_rbf, p_value_rbf = kernel_two_sample_test_nonuniform(self.YY0, self.YY1, self.e_input,
                                                                                   kernel_function='rbf',
                                                                                   gamma=1.0 / self.sigma2,
                                                                                   verbose=False,
                                                                                   iterations=self.perms
                                                                                   )
        return mmd2u_null_rbf, mmd2u_rbf



class baseline_test_gpu():
    def __init__(self,Y,e,T,permutations = 250,device='cuda:0'):
        if Y.shape[1]==1:
            self.YY0 = torch.from_numpy(Y[T == 0]).unsqueeze(-1).float().to(device)
            self.YY1 = torch.from_numpy(Y[T == 1]).unsqueeze(-1).float().to(device)
        else:
            self.YY0 = torch.from_numpy(Y[(T == 0).squeeze(),:]).float().to(device)
            self.YY1 = torch.from_numpy(Y[(T == 1).squeeze(),:]).float().to(device)
        # self.sigma2= np.median(pairwise_distances(self.YY0, self.YY1, metric='euclidean')) ** 2
        self.sigma2= general_ker_obj.get_median_ls(self.YY0,self.YY1)
        e_0 = e[T==0].float().to(device)
        e_1 = e[T==1].float().to(device)
        self.e_input = torch.cat([e_0,e_1],dim=0)
        self.perms = permutations


    def permutation_test(self):
        mmd2u_rbf, mmd2u_null_rbf, p_value_rbf = kernel_two_sample_test_nonuniform_gpu(X=self.YY0, Y=self.YY1,w= self.e_input,
                                                                                   kernel_function='rbf',
                                                                                   ls= self.sigma2,
                                                                                   verbose=False,
                                                                                   iterations=self.perms
                                                                                   )
        return mmd2u_null_rbf, mmd2u_rbf

class baseline_test_gpu_correct():
    def __init__(self,Y,e,T,permutations = 250,device='cuda:0'):
        if Y.shape[1]==1:
            self.YY0 = torch.from_numpy(Y[T == 0]).unsqueeze(-1).float().to(device)
            self.YY1 = torch.from_numpy(Y[T == 1]).unsqueeze(-1).float().to(device)
        else:
            self.YY0 = torch.from_numpy(Y[(T == 0).squeeze(),:]).float().to(device)
            self.YY1 = torch.from_numpy(Y[(T == 1).squeeze(),:]).float().to(device)
        # self.sigma2= np.median(pairwise_distances(self.YY0, self.YY1, metric='euclidean')) ** 2
        self.sigma2= general_ker_obj.get_median_ls(self.YY0,self.YY1)
        e_0 = e[T==0].float().to(device)
        e_1 = e[T==1].float().to(device)
        self.e_input = torch.cat([e_0,e_1],dim=0)
        self.perms = permutations

    def permutation_test(self):
        mmd2u_rbf, mmd2u_null_rbf, p_value_rbf = kernel_two_sample_test_nonuniform_gpu_correct(X=self.YY0, Y=self.YY1,w= self.e_input,
                                                                                   kernel_function='rbf',
                                                                                   ls= self.sigma2,
                                                                                   verbose=False,
                                                                                   iterations=self.perms
                                                                                   )
        return mmd2u_null_rbf, mmd2u_rbf





