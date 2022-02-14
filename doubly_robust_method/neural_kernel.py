import torch.nn

from doubly_robust_method.propensity_classifier import *
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
# class feature_map_kernel():
#     def __init__(self):

class feature_map(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 dropout=0.9,
                 transformation=torch.tanh,
                 output_dim=10,
                 ):
        super(feature_map, self).__init__()
        self.output_dim=output_dim
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout,output_dim)

    def identity_transform(self, x):
        return x

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout,output_dim):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential(*module_list)

        if len(layers_x)==1:
            self.final_layer = self.identity_transform
        else:
            self.final_layer = torch.nn.Linear(layers_x[-1],output_dim)

    def forward(self,x_cov,x_cat=[]):
        return self.final_layer(self.covariate_net((x_cov,x_cat)))

class inverse_hack(torch.nn.Module):
    def __init__(self,N,r):
        super(inverse_hack, self).__init__()
        self.C = torch.nn.Parameter(torch.randn(N,r))

    def forward(self,b):
        return self.C@(self.C.t()@b)

def congjuate_gradient_solver():
    pass

class neural_kme_model():
    def __init__(self,X_tr,Y_tr,T_tr,X_val,Y_val,T_val,
                 treatment_const,
                 neural_net_parameters,
                 r=0,
                 approximate_inverse=False,permutation_training=False,device='cuda:0'):
        self.X_tr = torch.from_numpy(X_tr[T_tr.squeeze()==treatment_const,:]).float().to(device)
        self.Y_tr = torch.from_numpy(Y_tr[T_tr.squeeze()==treatment_const]).float().to(device)
        self.X_val = torch.from_numpy(X_val[T_val.squeeze()==treatment_const,:]).float().to(device)
        self.Y_val = torch.from_numpy(Y_val[T_val.squeeze()==treatment_const]).float().to(device)

        self.n=X_tr.shape[0]
        self.eye = torch.eye(self.n).to(device)
        self.feature_map = feature_map(**neural_net_parameters)
        self.approximate_inverse = approximate_inverse
        self.permutation_training = permutation_training

        if self.permutation_training:
            self.inverse_hack = inverse_hack(self.n,r)

        ls_Y =general_ker_obj.get_median_ls(self.Y_tr, self.Y_tr)
        self.l = RBFKernel(self.Y_tr)
        self.l._set_lengthscale(ls_Y.item())
        self.L_tr = self.l.evaluate()
        self.L_val = self.l(self.Y_val,self.Y_val)
        self.L_cross = self.l(self.Y_tr,self.Y_val)

    def calculate_kernel(self,feature_map_1,feature_map_2,x1,x2):
        y_1 = feature_map_1(x1) #Nxd
        y_2 = feature_map_2(x2) #Mxd
        return y_1@(y_2.t())

    def kmvm_symmetric(self,feature_map_1,x1,b):
        y_1 = feature_map_1(x1) #Nxd
        return y_1@(y_1.t()@b)

    def calculate_tmp_term(self,X_1,X_2,X_3):
        pass
    def calculate_error(self,X_1,X_2,X_3,L,L_cross=None):
        if L_cross is None:
            L_cross = self.L_tr

        term_1 = ((tmp_calc.t()@self.L_tr) * tmp_calc.t()).sum(dim=1)
        term_2 =torch.diag(L)
        term_3 = -2*( tmp_calc*L_cross).sum(dim=0)
        error = term_1 + term_2 + term_3
        return error.mean()


    # def fit(self,its =10,patience=10):
    #     #TODO: Neural network regressor
    #     #Better cross val!
    #     self.best = np.inf
    #     self.patience=patience
    #     self.count=0
    #     self.kernel.ls.requires_grad=False
    #     self.kernel.lamb.requires_grad=False
    #     # self.opt= torch.optim.Adam(self.kernel.parameters(),lr=1e-2)
    #     list_of_lamb = np.linspace(0,5,20).tolist()
    #     for lamb in list_of_lamb:
    #         self.kernel.lamb[0] = lamb
    #         # for i in range(its):
    #         self.update_loop(lamb)
    #
    #         # for j in range(5):
    #         #     self.kernel.lamb.require_grad=False
    #         #     self.kernel.ls.require_grad=True
    #         #     self.update_loop(opt)
    #         #     if self.count>self.patience:
    #         #         return
    #         # for k in range(5):
    #         #     self.kernel.ls.require_grad = False
    #         #     self.kernel.lamb.require_grad=True
    #         #     self.update_loop(opt)
    #         #     if self.count>self.patience:
    #         #         return
    #     return
    # def get_middle_ker(self,X):
    #     middle_ker = self.kernel(self.X_tr, X)
    #     return middle_ker
    #
    # def calculate_validation_error(self,inv):
    #     with torch.no_grad():
    #         middle_ker = self.kernel(self.X_tr,self.X_val)
    #         tr_error_1 = self.calculate_error(inv,middle_ker,self.L_val,self.L_cross)
    #         return tr_error_1
    #
    # def calculate_error(self,inv,middle_ker,L,L_cross=None):
    #     if L_cross is None:
    #         L_cross = self.L_tr
    #     tmp_calc = inv@middle_ker
    #     term_1 = ((tmp_calc.t()@self.L_tr) * tmp_calc.t()).sum(dim=1)
    #     term_2 =torch.diag(L)
    #     term_3 = -2*( tmp_calc*L_cross).sum(dim=0)
    #     error = term_1 + term_2 + term_3
    #     return error.mean()
    #
    # #TODO: needs fixing wow
    # def get_psi_part(self, x_te, T_te):
    #     with torch.no_grad():
    #         middle_ker = self.kernel(x_te,self.X_tr)
    #         tmp_val = (T_te.t()@middle_ker)@self.inv
    #     return tmp_val
    #
    # def get_psi_square_term(self, tmp_val):
    #     return (tmp_val.t()@self.L_tr)@tmp_val
    #
    # def get_psi_cross_term(self, psi_part,Y_te,T_te):
    #     b_term = self.l(self.Y_tr,Y_te)@T_te
    #     return psi_part.t()@b_term
    # def get_weird_cross_term(self,left,right,other_Y_tr):
    #     b_term = self.l(self.Y_tr,other_Y_tr)@right
    #     return left.t()@b_term
    #



