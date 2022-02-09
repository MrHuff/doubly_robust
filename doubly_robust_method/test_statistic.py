import torch
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
general_ker_obj =Kernel()

class counterfactual_me_test():
    def __init__(self,X,Y,e,T,kme_1,kme_0,permute_e=False,permutations=250,device='cuda:0'):
        self.permutations = permutations
        self.X=torch.from_numpy(X).float().to(device)
        self.n= Y.shape[0]
        self.Y = torch.from_numpy(Y).float().to(device)
        self.e =e.to(device)
        self.T_1 = torch.from_numpy(T).float().to(device)
        self.T_0 =1.-self.T_1
        self.kme_1=kme_1
        self.kme_0=kme_0
        self.permute_e = permute_e
        self.create_all_weights(self.e)
        self.kernel = RBFKernel(self.Y).to(device)
        self.ls =general_ker_obj.get_median_ls(self.Y, self.Y)
        self.kernel._set_lengthscale(self.ls)
        self.L = self.kernel.evaluate()
        self.calculate_psi()
        self.ref_stat = self.calculate_test_statistic(self.L,self.e)

    def calculate_psi(self):

        self.psi_1 = self.kme_1.get_psi_part(self.X, self.psi_1_weight).t()
        self.psi_0 = self.kme_0.get_psi_part(self.X, self.psi_0_weight).t()

    def calculate_test_statistic(self,L,e=None):
        if self.permute_e:
            self.create_all_weights(e)
            self.calculate_psi()
        T_1_L_test = L@self.T_1_weight
        T_0_L_test = L@self.T_0_weight
        term_1 = T_1_L_test.t()@self.T_1_weight
        term_2 = self.kme_1.get_psi_square_term(self.psi_1)
        term_3 = -2*self.kme_1.get_psi_cross_term(self.psi_1,self.Y,self.T_1_weight)
        term_4 = T_0_L_test.t()@self.T_0_weight
        term_5 = self.kme_0.get_psi_square_term(self.psi_0)
        term_6 = -2*self.kme_0.get_psi_cross_term(self.psi_0,self.Y,self.T_0_weight)
        term_7 = -2*T_1_L_test.t()@T_0_L_test
        term_8 = 2*self.kme_0.get_psi_cross_term(self.psi_0,self.Y,self.T_1_weight)
        term_9 = 2*self.kme_1.get_psi_cross_term(self.psi_1,self.Y,self.T_0_weight)
        term_10 = -2*self.kme_0.get_weird_cross_term(left=self.psi_0,right=self.psi_1,other_Y_tr=self.kme_1.Y_tr)
        tot = term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8+term_9+term_10
        return 1/self.n**2 * tot
        # return self.omega.t()@(L@self.omega).item()

    def create_all_weights(self,e):
        self.psi_0_weight=(e-self.T_1)/(1.-e)
        self.psi_1_weight=(self.T_1-e)/(e)
        self.T_1_weight=self.T_1/e
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
            perm_stats.append(tst.item())
        return perm_stats,self.ref_stat.item()

if __name__ == '__main__':
    pass