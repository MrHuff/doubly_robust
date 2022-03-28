import pandas as pd
import os
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_perm(s,n,m,X_in,Y_in,Z):
    if not os.path.exists(f'datasets/inspire_{m}'):
        os.makedirs(f'datasets/inspire_{m}')
    if not os.path.exists(f'datasets/inspire_{m}_null'):
        os.makedirs(f'datasets/inspire_{m}_null')
    np.random.seed(s)
    perm_vec = np.random.permutation(n)[:m]
    T= X_in[perm_vec][:,np.newaxis]
    Y=Y_in[perm_vec][:,np.newaxis]
    X=Z[perm_vec,:]
    with open(f'datasets/inspire_{m}/job_{s}.pickle', 'wb') as handle:
        pickle.dump({'seed': s, 'T': T, 'Y': Y, 'X': X, 'W': T}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'datasets/inspire_{m}_null/job_{s}.pickle', 'wb') as handle:
        pickle.dump({'seed': s, 'T': T, 'Y': np.random.randn(*Y.shape), 'X': X, 'W': T}, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    confounders = ['Age_c1','MOrp_i1','Rural_i1','POrp_i1','Food_i12','Pov_i12','PosP_i12','ScMe_i12','LTFU_i12','ScFr_i12','HIV_i1','HHsiz_c1','House_i1','Sex_i1','Prov_i1','PosP_c12']
    outcomes = [
        'AbPhy_i2',
        'AbEmo_i2',
        'AbSx_i2',
        'ViPerp_i2'
    ]
    df = pd.read_stata('20210608 Plos Medicine INSPIRE.dta')
    T= df['MonP_c12'].apply(lambda x: x<24).values.astype('float')
    X=df[confounders]
    X=pd.get_dummies(X,columns=['MOrp_i1','Rural_i1','POrp_i1','Food_i12','Pov_i12','PosP_i12','ScMe_i12','LTFU_i12','ScFr_i12','HIV_i1','House_i1','Sex_i1','Prov_i1'])
    X=X.drop(['Sex_i1_0. Boy','House_i1_0. No','HIV_i1_0. No','MOrp_i1_0. No','POrp_i1_0. No','Food_i12_0. No','Pov_i12_0. No','PosP_i12_0. No','ScMe_i12_0. No','LTFU_i12_0. No','ScFr_i12_0. No'],axis=1).values
    Y = df[outcomes]
    Y=pd.get_dummies(Y)
    Y=Y.drop(['AbPhy_i2_0. No','AbEmo_i2_0. No','AbSx_i2_0. No','ViPerp_i2_0. No'],axis=1).values
    n=Y.shape[0]
    for s in range(100):
        get_perm(s=s,n=n,m=1000,X_in=T,Y_in=Y,Z=X)

