import torch
from kernels import *


class counterfactual_me_test():
    def __init__(self,Y,e,T,kernel_type ='rbf',ls=1.0):
        self.Y = Y
        self.e =e
        self.T_1 = T
        self.T_0 =1-T
        
        #kx is 0, ky is 1 in their code



if __name__ == '__main__':
    pass