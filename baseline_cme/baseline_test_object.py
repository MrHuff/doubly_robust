import numpy as np

from baseline_cme.kernel_two_sample_test_nonuniform import *
from sklearn.metrics import pairwise_distances

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






