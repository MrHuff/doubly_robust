import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from generate_job_parameters import load_obj,linear
import os
import re
import numpy as np
import scipy.stats as st
from pylatex import Document, Section, Figure, SubFigure, NoEscape,Command
import itertools
from pylatex.base_classes import Environment
from pylatex.package import Package
sns.set()

def calc_error_bars(power_val,alpha,num_samples):
    z = st.norm.ppf(1.-alpha/2.)
    up = power_val+z*(power_val*(1-power_val)/num_samples)**0.5
    down = power_val-z*(power_val*(1-power_val)/num_samples)**0.5
    z_vec = np.ones_like(power_val)*z*(power_val*(1-power_val)/num_samples)**0.5
    return up,down,z_vec


def extract_bdn(str):
    el_list= re.findall(r"[-+]?(?:\d*\.\d+|\d+)",str)
    return el_list[0],el_list[1],el_list[2]

def get_method(tp,neural_cme,double_cme,oracle_weights,method):
    method_str=''
    if oracle_weights:
        method_str+='ow-'
    if tp:
        method_str+='ep-'
    method.replace('_','-')
    method_str+=method.replace('_','')
    if neural_cme:
        method_str+='-nncme'
    if double_cme:
        method_str+='-dcme'
    return method_str


def get_job_df(job_path):
    jobs = os.listdir(job_path)
    data = []
    columns = ['tp','neural_cme','double_estimate_kme','oracle_weights','method','b','D','n','pval_001','pval_005','pval_01','KS_pval','KS_stat']
    for j in jobs:
        experiment_params = load_obj(j, folder=f'{job_path}/')
        method=experiment_params['test_type']
        ow=experiment_params['training_params']['oracle_weights']
        dek=experiment_params['training_params']['double_estimate_kme']
        tp=experiment_params['training_params']['epochs']==100
        ncme=experiment_params['training_params']['neural_cme']
        b,d,n=extract_bdn(j)
        df= pd.read_csv(experiment_params['experiment_save_path']+f'/final_res.csv',index_col=0).values.tolist()[0]
        row = [tp,ncme,dek,ow,method,b,d,n]+df
        data.append(row)
    job_df = pd.DataFrame(data,columns=columns)
    full_method = []
    for i,r in job_df.iterrows():
        d=r[['tp','neural_cme','double_estimate_kme','oracle_weights','method']].tolist()
        m_str = get_method(*d)
        full_method.append(m_str)
    job_df['mname']=full_method
    return job_df

class subfigure(Environment):
    """A class to wrap LaTeX's alltt environment."""
    packages = [Package('subcaption')]
    escape = False
    content_separator = "\n"
    _repr_attributes_mapping = {
        'position': 'options',
        'width': 'arguments',
    }

    def __init__(self, position=NoEscape(r'H'),width=NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.
        """

        super().__init__(options=position,arguments=width, **kwargs)

dict_method = {''}
def plot_2_est_weights(dir,df,d_list,methods,nlist,tname='pval_005'):

    if not os.path.exists(dir):
        os.makedirs(dir)
    for d in d_list:
        subset = df[df['D']==d].sort_values(['b'])
        for n in nlist:
            subset_2 = subset[subset['n'] == n]
            for col_index,method in enumerate(methods):
                subset_3 = subset_2[subset_2['mname']==method]
                a,b,e = calc_error_bars(subset_3[tname],alpha=0.05,num_samples=100)
                plt.plot('b',tname,data=subset_3,linestyle='--', marker='o',label=rf'{method}')
                # plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='-',label=rf'{format_string}',c = col_dict[method])

                plt.fill_between(subset_3['b'], a, b, alpha=0.1)
                # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1,color=col_dict[method])
            plt.hlines(0.05, 0, subset_2['b'].max())
            plt.legend(prop={'size': 10})
            plt.xlabel(r'$\beta_{XY}$')
            plt.ylabel('Power')
            plt.savefig(f'{dir}/figure_{d}_{n}.png',bbox_inches = 'tight',
        pad_inches = 0.05)
            plt.clf()


def generate_tex_plot(nlist):
    doc = Document(default_filepath=dir)
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate([1, 3, 15, 50]):
                if i == 0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(r'0.24\linewidth'))):
                    name = f'$d_Z={n}$'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
        counter = 0
        for idx, (i, j) in enumerate(itertools.product(nlist, [1, 3, 15, 50])):
            if idx % 4 == 0:
                name = f'$n={i}$'
                string_append = r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % name + '%\n'
            p = f'{dir}/figure_{j}_{i}.png'
            string_append += r'\includegraphics[width=0.24\linewidth]{%s}' % p + '%\n'
            counter += 1
            if counter == 4:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter = 0
    doc.generate_tex()

if __name__ == '__main__':
    job_path='recreate_krik_big'
    df = get_job_df(job_path)
    plot_2_est_weights(dir='krik_big_2',df=df,
                       d_list=df['D'].unique().tolist(),
                       methods=df['mname'].unique().tolist(),
                       nlist=df['n'].unique().tolist())



