import numpy as np

from kernels import *
import tqdm
import torch
class kme_model(torch.nn.Module):
    def __init__(self,X_tr,Y_tr,T_tr,X_val,Y_val,T_val,device='cuda:0'):
        super(kme_model, self).__init__()
        self.device=device
        self.X_tr=X_tr.to(device)
        self.Y_tr = Y_tr.to(device)
        self.T_1= T_tr.to(device)
        self.T_0 =1-self.T_1
        self.n=X_tr.shape[0]
        self.w_t_1_tr = self.T_1@(self.T_1.t())
        self.w_t_0_tr = self.T_0@(self.T_0.t())


        self.X_val = X_val.to(device)
        self.Y_val = Y_val.to(device)
        self.T_1_val = T_val.to(device)
        self.T_0_val = 1 - self.T_1_val

        self.lamb = torch.nn.Parameter(torch.ones(1,1)*1e-3,requires_grad=True).to(device)
        self.ls = torch.nn.Parameter(torch.ones(1,1),requires_grad=True).to(device)
        self.eye = torch.eye(self.n).to(device)

        self.kernel = RBFKernel(self.X_tr)
        self.kernel._set_lengthscale(self.ls)

    def construct_inverse(self):
        middle_ker=self.kernel.evaluate()
        self.inv = torch.inverse(middle_ker+self.eye*self.lamb)
        return self.inv,middle_ker

    def calc_r2(self,val_error,y):
        return 1.-val_error/y.var()

    def update_loop(self,opt):
        inv, middle_ker = self.construct_inverse()
        self.inv_t_1 = self.inv * self.w_t_1_tr
        self.inv_t_0 = self.inv * self.w_t_0_tr
        total_error = self.calculate_training_error(middle_ker)
        opt.zero_grad()
        total_error.backwards()
        opt.step()
        val_error = self.calculate_validation_error()
        val_r2 = self.calc_r2(val_error, self.Y_val)

        if val_r2.item()>self.best:
            self.best =val_r2.item()
            self.count=0
        else:
            self.count+=1

        print(val_r2.item())
    def fit(self,its =10,patience=10):
        self.best = -np.inf
        self.patience=patience
        self.count=0
        opt= torch.optim.Adam(params=(self.lamb,self.ls),lr=1e-2)
        for i in tqdm.tqdm(range(its)):
            for i in range(5):
                self.lamb.require_grad=False
                self.ls.require_grad=True
                self.update_loop(opt)
                if self.count>self.patience:
                    return
            for j in range(5):
                self.ls.require_grad = False
                self.lamb.require_grad=True
                self.update_loop(opt)
                if self.count>self.patience:
                    return
        return


    def calculate_validation_error(self):
        with torch.no_grad():
            middle_ker = self.kernel(self.X_tr,self.X_val)
            y1_preds = (self.inv_t_1) @ ((middle_ker @ self.T_1_val).t() @ self.Y_val).t()
            y0_preds = (self.inv_t_0) @ ((middle_ker @ self.T_0_val).t() @ self.Y_val).t()
            tr_error_1 = torch.sum((y1_preds - self.Y_val * self.T_1_val) ** 2)
            tr_error_0 = torch.sum((y0_preds - self.Y_val * self.T_0_val) ** 2)
            total_error = (tr_error_1 + tr_error_0) /self.X_val.shape[0]
            return total_error

    def calculate_training_error(self,middle_ker):
        y1_preds = (self.inv_t_1) @ ((middle_ker @ self.T_1).t() @ self.Y_tr).t()
        y0_preds = (self.inv_t_0) @ ((middle_ker @ self.T_0).t() @ self.Y_tr).t()
        tr_error_1 = torch.sum((y1_preds - self.Y_tr * self.T_1) ** 2)
        tr_error_0 = torch.sum((y0_preds - self.Y_tr * self.T_0) ** 2)
        total_error = (tr_error_1 + tr_error_0) / self.n
        return total_error
    def get_psi_1(self,x_te,T_te):
        middle_ker = self.kernel(self.X_tr,x_te)
        return self.inv_t_1@(middle_ker@T_te)

    def get_psi_0(self, x_te, T_te):
        middle_ker = self.kernel(self.X_tr,x_te)
        return self.inv_t_0@(middle_ker@T_te)







