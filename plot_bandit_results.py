import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import os

from scipy.stats.mstats import winsorize
sns.set()
plt.switch_backend('agg')
# plt.rcParams['font.size'] = 20
def winsorized_std(data, percentile):
    std = np.std(winsorize(data, (0, percentile)).data)
    return std

def winsorized_mean(data, percentile):
    mean = np.mean(winsorize(data, (0, percentile)).data)
    return mean

# combine the result files
print(sns.color_palette())
def produce_plot(i,ax,keyword_1,keyword,cov_name):
    df_list = [pd.read_csv(filename) for filename in glob.glob(os.path.join(f'{keyword_1}_report/results','*.csv'))]

    prelim_result = pd.concat(df_list)
    prelim_result.sort_values(by=[keyword], ascending=True)

    prelim_result = prelim_result[[c for c in prelim_result.columns if 'error' in c] + [keyword]]
    prelim_result.columns = prelim_result.columns.str.replace("_square_error", "")

    estimator_cols = list(filter(lambda x: 'estimator' in x, prelim_result))

    # compute the statistics and plot the results
    winsorized_df = pd.DataFrame()
    for cond, cond_df in prelim_result.groupby(keyword):
        for e in estimator_cols:
            cond_df[e] = winsorize(cond_df[e], (0, 0.1))

        cond_df[keyword] = cond
        winsorized_df = winsorized_df.append(cond_df)

    # prelim_result = prelim_result.query("multiplier < -0.2")
    final_df = pd.melt(winsorized_df, id_vars=[keyword], var_name="estimator", value_name="MSE")

    # plot the results
    sns.lineplot(ax=ax,x=keyword, y="MSE", hue="estimator", data=final_df,palette=[(0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])

    ax.set_yscale('log')
    ax.set_ylabel("MSE",fontsize=18)
    ax.set_xlabel(cov_name,fontsize=18)

    # xticks = ['%0.1f' % x for x in np.unique(final_df[keyword])]
    # ax.xaxis.set_major_locator(plt.FixedLocator(range(len(xticks))))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter(xticks))
    if i==5:
        leg_handles = ax.get_legend_handles_labels()[0]
        # ax.legend(leg_handles, ['CME', 'Direct', 'DR', 'Slate', 'IPS'], title='Estimator')
        ax.legend(leg_handles, ['DR-CFME','CFME'], title='Estimator')

    # plt.tight_layout()
    # plt.savefig(f'{keyword_1}_result.pdf',bbox_inches = 'tight',
    #         pad_inches = 0.05)
    # plt.clf()

if __name__ == '__main__':
    # keywords_1 = ['context_dim','domain_shift','recommendation_size','item_size','sample_size','user_size']
    keywords_1 = ['domain_shift','context_dim','item_size','user_size','recommendation_size','sample_size']
    # keywords = ['context_dim','multiplier','n_reco','num_items','sample_size','num_users']
    keywords = ['multiplier','context_dim','num_items','num_users','n_reco','sample_size']
    # cov_names  = ['Dimension of Covariates','Multiplier','Number of Recommended Items','Number of Items','Number of Observations','Number of Users']
    cov_names  = ['Multiplier','Dimension of Covariates','Number of Items','Number of Users','Number of Recommended Items','Number of Observations']
    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(18, 6))

    for i,(a,b,c) in enumerate(zip(keywords_1,keywords,cov_names)):
        print(i%2,i%3)
        ax = axes[i%2,i%3]
        produce_plot(i,ax,a,b,c)
        # if i < 5:
        ax.legend([], [], frameon=False)

    plt.tight_layout()
    plt.savefig(f'big_result.pdf',bbox_inches = 'tight',
            pad_inches = 0.05)
    plt.clf()