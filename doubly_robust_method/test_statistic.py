import torch
from kernels import *
from kme import *
general_ker_obj =Kernel()
class counterfactual_me_test():
    def __init__(self,X,Y,e,T,kme_1,kme_0,permute_e=False,permutations=250,device='cuda:0'):
        self.permutations = permutations
        self.X=X.to(device)
        self.n= Y.shape[0]
        self.Y = Y.to(device)
        self.e =e.to(device)
        self.T_1 = T.to(device)
        self.T_0 =1-T
        self.kme_1=kme_1
        self.kme_0=kme_0
        self.permute_e = permute_e
        self.create_all_weights(self.e)
        self.kernel = RBFKernel(self.Y).to(device)
        self.ls =general_ker_obj.get_median_ls(Y, Y)
        self.kernel._set_lengthscale(self.ls)
        self.L = self.kernel.evaluate()
        self.ref_stat = self.calculate_test_statistic(self.L,self.e)

    def calculate_psi_omega(self):
        self.psi_1 = self.kme_1.get_psi(self.X, self.psi_1_weight)
        self.psi_0 = self.kme_0.get_psi(self.X, self.psi_0_weight)
        self.omega = self.psi_1+self.T_0_weight-self.psi_0-self.T_1_weight

    def calculate_test_statistic(self,L,e=None):
        if self.permute_e:
            self.create_all_weights(e)
            self.calculate_psi_omega()
        return self.omega.t()@(L@self.omega).item()

    def create_all_weights(self,e):
        self.psi_0_weight=(e-self.T_1)/(1.-e)
        self.psi_1_weight=(self.T_1-e)/(e)
        self.T_1_weight=e/self.T_1
        self.T_0_weight = self.T_0/(1.-e)

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_test(self):
        perm_stats=[]
        for i in range(self.permutations):
            perm_L,idx = self.get_permuted2d(self.L)
            if self.permute_e:
                e=self.e[idx]
            else:
                e=self.e
            tst = self.calculate_test_statistic(perm_L,e)
            perm_stats.append(tst)
        return perm_stats,self.ref_stat

if __name__ == '__main__':
    pass