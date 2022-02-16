import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':
    load_string = 'dr_exp/derangement_prop_doubly_robust_0.0/big_df.csv'
    df = pd.read_csv(load_string,index_col=0)
    hist_data  = df.values
    ind = 99
    test_stats_perm = hist_data[ind,3:]
    ref_stat = hist_data[ind,2]
    print(ref_stat)
    print(test_stats_perm)
    plt.hist(test_stats_perm,normed=True)
    plt.axvline(x=ref_stat, color='r',ymin=0.0,ymax=1.0)

    # plt.hist([ref_stat],color='red')
    plt.show()

    # df.

