import numpy as np

from data_generation.data_generation import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()

def debug_plot_weights(T,w_true):
    df = pd.DataFrame(np.concatenate([T,w_true],axis=1),columns=['T','prob'])
    sns.histplot(data=df,x='prob',hue='T',bins=50)
    plt.show()
def debug_plot_treatments(T,Y):
    df = pd.DataFrame(np.concatenate([T,Y],axis=1),columns=['T','Y'])
    sns.histplot(data=df,x='Y',hue='T',bins=50)
    plt.show()


if __name__ == '__main__':
    s=0
    D=5
    b=0.0
    base_config = {'seed': s,
                   'ns': 1000,
                   'd': D,
                   'alpha_vec': np.array([0.05, 0.04, 0.03, 0.02, 0.01]) * 35,  # Treatment assignment
                   # the thing just blows up regardless of what you do?!
                   # np.array([0.05,0.04,0.03,0.02,0.01]),#np.random.randn(5)*0.05, #np.array([0.05,0.04,0.03,0.02,0.01]),
                   'alpha_0': 0.05,  # 0.05,
                   'beta_vec': np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.05,  # Confounding
                   'noise_var': 0.1,
                   'b': b
                   }
    T, Y, X, w_true = case_1(**base_config)
    debug_plot_weights(T,w_true)
    debug_plot_treatments(T,Y)
    #DONT PERMUTE THE E WEIGHTS!





