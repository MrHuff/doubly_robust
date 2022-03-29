import os

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

types = {'adequacy': 'cat',
 'alcohol': 'bin',
 'anemia': 'bin',
 'birattnd': 'cat',
 'birmon': 'cyc',
 'bord': 'bin',
 'brstate': 'cat',
 'brstate_reg': 'cat',
 'cardiac': 'bin',
 'chyper': 'bin',
 'cigar6': 'cat',
 'crace': 'cat',
 'csex': 'bin',
 'data_year': 'cat',
 'dfageq': 'cat',
 'diabetes': 'bin',
 'dlivord_min': 'ord',
 'dmar': 'bin',
 'drink5': 'cat',
 'dtotord_min': 'ord',
 'eclamp': 'bin',
 'feduc6': 'cat',
 'frace': 'cat',
 'gestat10': 'cat',
 'hemo': 'bin',
 'herpes': 'bin',
 'hydra': 'bin',
 'incervix': 'bin',
 'infant_id': 'index do not use',
 'lung': 'bin',
 'mager8': 'cat',
 'meduc6': 'cat',
 'mplbir': 'cat',
 'mpre5': 'cat',
 'mrace': 'cat',
 'nprevistq': 'cat',
 'orfath': 'cat',
 'ormoth': 'cat',
 'othermr': 'bin',
 'phyper': 'bin',
 'pldel': 'cat',
 'pre4000': 'bin',
 'preterm': 'bin',
 'renal': 'bin',
 'rh': 'bin',
 'stoccfipb': 'cat',
 'stoccfipb_reg': 'cat',
 'tobacco': 'bin',
 'uterine': 'bin'}



X = pd.read_csv("TWINS/twin_pairs_T_3years_samesex.csv",index_col=[0])
Y = pd.read_csv("TWINS/twin_pairs_Y_3years_samesex.csv",index_col=[0])
Z = pd.read_csv("TWINS/twin_pairs_X_3years_samesex.csv",index_col=[0])

Z = Z.drop(['infant_id_1','Unnamed: 0','infant_id_0'],axis=1)
Z = Z.dropna()
rows_to_keep = Z.index.tolist()

X = X.iloc[rows_to_keep,:]
X = (X['dbirwt_1'].values-X['dbirwt_0'].values)>=250
Y = Y.iloc[rows_to_keep,:]
Y = (Y['mort_0']-Y['mort_1']).values

def get_perm(s,n,m,X_in,Y_in,Z):
    if not os.path.exists(f'datasets/twins_{m}'):
        os.makedirs(f'datasets/twins_{m}')
    if not os.path.exists(f'datasets/twins_{m}_null'):
        os.makedirs(f'datasets/twins_{m}_null')
    np.random.seed(s)
    perm_vec = np.random.permutation(n)[:m]
    T= X_in[perm_vec][:,np.newaxis]
    Y=Y_in[perm_vec]
    X=Z[perm_vec,:]
    with open(f'datasets/twins_{m}/job_{s}.pickle', 'wb') as handle:
        pickle.dump({'seed': s, 'T': T, 'Y': Y, 'X': X, 'W': T}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'datasets/twins_{m}_null/job_{s}.pickle', 'wb') as handle:
        pickle.dump({'seed': s, 'T': T, 'Y': np.random.randn(*Y.shape), 'X': X, 'W': T}, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    cat_cols_names = []
    for i,j in types.items():
        if j =='cat':
            cat_cols_names.append(i)
    col_counts = []
    col_stats_list = []
    col_index_list = [False]*Z.shape[1]
    for cat_col in cat_cols_names:
        col_index_list[Z.columns.get_loc(cat_col)]=True
    cat_cols = Z.iloc[:,col_index_list]

    for i in range(cat_cols.shape[1]):
        col_stats = cat_cols.iloc[:,i].unique().tolist()
        col_stats_list.append(col_stats)
        col_counts.append(len(col_stats))
    Z = pd.get_dummies(Z,columns=cat_cols_names)
    scaler_1 = StandardScaler()
    Z = Z.values
    Z = scaler_1.fit_transform(Z)
    scaler_2 = StandardScaler()
    Y = scaler_2.fit_transform(Y[:,np.newaxis])
    other_data = {'indicator': col_index_list, 'index_lists': col_stats_list}
    for m in [2500,5000]:
        for s in range(100):
            get_perm(s=s,n=X.shape[0],m=m,X_in=X,Y_in=Y,Z=Z)

    with open(f'datasets/twins_cat_col_data.pickle', 'wb') as handle:
        pickle.dump(other_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # print(X.sum())
    # print((~X).sum())
    # plt.hist(X,bins=2)
    # plt.show()
    # plt.clf()
    # X = (X-X.mean(0))/X.std(0)
    # plt.hist(X.values)
    # plt.show()








