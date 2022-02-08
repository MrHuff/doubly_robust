import numpy as np

from doubly_robust_method.kernels import *
import tqdm
import torch
class kme_model(torch.nn.Module):
    def __init__(self,X_tr,Y_tr,T_tr,X_val,Y_val,T_val,treatment_const,device='cuda:0'):
        super(kme_model, self).__init__()
        self.device=device

        # self.T= T_tr.to(device)
        # self.T_0 =1-self.T_1
        self.X_tr = torch.from_numpy(X_tr[T_tr.squeeze()==treatment_const,:]).to(device)
        self.Y_tr = torch.from_numpy(Y_tr[T_tr.squeeze()==treatment_const]).to(device)

        self.n=X_tr.shape[0]

        self.X_val = torch.from_numpy(X_val[T_val.squeeze()==treatment_const,:]).to(device)
        self.Y_val = torch.from_numpy(Y_val[T_val.squeeze()==treatment_const]).to(device)

        self.lamb = torch.nn.Parameter(torch.tensor([1e-3]).float(),requires_grad=True).to(device)
        self.ls = torch.nn.Parameter(torch.tensor([1.]).float(),requires_grad=True).to(device)
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
        self.inv, middle_ker = self.construct_inverse()
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
        opt= torch.optim.Adam(self.kernel.parameters(),lr=1e-2)
        for i in tqdm.tqdm(range(its)):
            for j in range(5):
                self.lamb.require_grad=False
                self.ls.require_grad=True
                self.update_loop(opt)
                if self.count>self.patience:
                    return
            for k in range(5):
                self.ls.require_grad = False
                self.lamb.require_grad=True
                self.update_loop(opt)
                if self.count>self.patience:
                    return
        return

    def calculate_validation_error(self):
        with torch.no_grad():
            middle_ker = self.kernel(self.X_tr,self.X_val)
            y1_preds = self.inv @ (middle_ker.t() @ self.Y_val).t()
            tr_error_1 = torch.mean((y1_preds - self.Y_val) ** 2)
            return tr_error_1

    def calculate_training_error(self,middle_ker):
        y1_preds = (self.inv) @ (middle_ker.t() @ self.Y_tr).t()
        tr_error_1 = torch.mean((y1_preds - self.Y_tr) ** 2)
        return tr_error_1

    def get_psi(self, x_te, T_te):
        middle_ker = self.kernel(self.X_tr,x_te)
        return self.inv@(middle_ker@T_te)







