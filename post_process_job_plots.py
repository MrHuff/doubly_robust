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

def extract_bdn_real(str):
    el_list= re.findall(r"[-+]?(?:\d*\.\d+|\d+)",str)
    return el_list[0]
def extract_dataset_method(str):
    pass

def get_method(tp,neural_cme,double_cme,oracle_weights,method):
    if method in ['doubleml','vanilla_dr','gformula','tmle','ipw','cf','bart','wmmd']:
        method.replace('_', '-')
        return method
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
    columns = ['dataset','tp','neural_cme','double_estimate_kme','oracle_weights','method','b','D','n','pval_001','pval_005','pval_01','KS_pval','KS_stat']
    for j in jobs:
        try:
            experiment_params = load_obj(j, folder=f'{job_path}/')
            method=experiment_params['test_type']
            ow=experiment_params['training_params']['oracle_weights']
            dek=experiment_params['training_params']['double_estimate_kme']
            tp=experiment_params['training_params']['epochs']==100
            ncme=experiment_params['training_params']['neural_cme']
            b,d,n=extract_bdn(j)
            ds = j.split(method)[0][0:-1]
            df= pd.read_csv(experiment_params['experiment_save_path']+f'/final_res.csv',index_col=0).values.tolist()[0]
            row = [ds,tp,ncme,dek,ow,method,b,d,n]+df
            data.append(row)
        except Exception as e:
            print(e)
    job_df = pd.DataFrame(data,columns=columns)
    full_method = []
    for i,r in job_df.iterrows():
        d=r[['tp','neural_cme','double_estimate_kme','oracle_weights','method']].tolist()
        m_str = get_method(*d)
        full_method.append(m_str)
    job_df['mname']=full_method
    return job_df

def get_job_df_real(job_path):
    jobs = os.listdir(job_path)
    data = []
    columns = ['dataset','tp','neural_cme','double_estimate_kme','oracle_weights','method','n','final_res_path','pval_001','pval_005','pval_01','KS_pval','KS_stat',]
    for j in jobs:
        try:
            experiment_params = load_obj(j, folder=f'{job_path}/')
            method=experiment_params['test_type']
            ow=experiment_params['training_params']['oracle_weights']
            dek=experiment_params['training_params']['double_estimate_kme']
            tp=experiment_params['training_params']['epochs']==100
            ncme=experiment_params['training_params']['neural_cme']
            n=extract_bdn_real(j)
            ds = j.split(method)[0][0:-1]
            df= pd.read_csv(experiment_params['experiment_save_path']+f'/final_res.csv',index_col=0).values.tolist()[0]
            fn_path = experiment_params['experiment_save_path'] + f'/big_df.csv'
            row = [ds,tp,ncme,dek,ow,method,n,fn_path]+df
            data.append(row)
        except Exception as e:
            print(e)
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
def plot_2_est_weights(dir,big_df,d_list,methods,nlist,data_list,tname='pval_005'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for dataset in data_list:
        df = big_df[big_df['dataset']==dataset]
        for d in d_list:
            subset = df[df['D']==d]
            for n in nlist:
                subset_2 = subset[subset['n'] == n]
                for col_index,method in enumerate(methods):
                    try:
                        subset_3 = subset_2[subset_2['mname']==method].sort_values(['b'],ascending=True).reset_index()
                        a,b,e = calc_error_bars(subset_3[tname],alpha=0.05,num_samples=100)
                        plt.plot('b',tname,data=subset_3,linestyle='--', marker='o',label=rf'{method}')
                        # plt.plot('beta_xy','p_a=0.05',data=subset_3,linestyle='-',label=rf'{format_string}',c = col_dict[method])

                        plt.fill_between(subset_3['b'], a, b, alpha=0.1)
                    except Exception as e:
                        print('whoopsie')
                    # plt.fill_between(subset_3['beta_xy'], a, b, alpha=0.1,color=col_dict[method])

                # plt.figure(figsize=(15,15))
                plt.hlines(0.05, 0,subset_3['b'].max())
                plt.legend(prop={'size': 10})
                plt.xlabel(r'$\beta_{TY}$')
                plt.ylabel('Power')
                plt.savefig(f'{dir}/{dataset}_figure_{d}_{n}.png',bbox_inches = 'tight',
            pad_inches = 0.05)
                plt.clf()

def gpu_post_process(fold_name,df):
    mask_ow = df['mname'].apply(lambda x: 'ow' in x).values
    df_ow =df[mask_ow]
    df_not_ow =df[~mask_ow]
    # df_ow = df[df['']]
    plot_2_est_weights(dir=f'{fold_name}_ow',big_df=df_ow,
                       d_list=df_ow['D'].unique().tolist(),
                       methods=df_ow['mname'].unique().tolist(),
                       nlist=df_ow['n'].unique().tolist(),
                       data_list=df_ow['dataset'].unique().tolist()
                       )
    plot_2_est_weights(dir=f'{fold_name}_not_ow',big_df=df_not_ow,
                       d_list=df_not_ow['D'].unique().tolist(),
                       methods=df_not_ow['mname'].unique().tolist(),
                       nlist=df_not_ow['n'].unique().tolist(),
                       data_list=df_not_ow['dataset'].unique().tolist()
                       )
def plot_1(df):
    list_of_stuff=['ep-baseline','ep-doublyrobust-dcme']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    plot_2_est_weights(dir=f'plot_1',big_df=subset_df,
                       d_list=subset_df['D'].unique().tolist(),
                       methods=subset_df['mname'].unique().tolist(),
                       nlist=subset_df['n'].unique().tolist(),
                       data_list=subset_df['dataset'].unique().tolist()
                       )
def plot_1a(df):
    list_of_stuff=['ep-doublyrobustcorrect-dcme','ep-doublyrobust-dcme']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    plot_2_est_weights(dir=f'plot_1a',big_df=subset_df,
                       d_list=subset_df['D'].unique().tolist(),
                       methods=subset_df['mname'].unique().tolist(),
                       nlist=subset_df['n'].unique().tolist(),
                       data_list=subset_df['dataset'].unique().tolist()
                       )
def plot_1b(df):
    list_of_stuff=['baseline','doublyrobust-dcme']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    plot_2_est_weights(dir=f'plot_1b',big_df=subset_df,
                       d_list=subset_df['D'].unique().tolist(),
                       methods=subset_df['mname'].unique().tolist(),
                       nlist=subset_df['n'].unique().tolist(),
                       data_list=subset_df['dataset'].unique().tolist()
                       )
def plot_2(df):
    list_of_stuff=['ep-doublyrobust-dcme','ep-doublyrobustcorrect-dcme']+['doubleml','vanilla-dr','gformula','tmle','ipw','cf','bart','wmmd']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    plot_2_est_weights(dir=f'plot_2',big_df=subset_df,
                       d_list=subset_df['D'].unique().tolist(),
                       methods=subset_df['mname'].unique().tolist(),
                       nlist=subset_df['n'].unique().tolist(),
                       data_list=subset_df['dataset'].unique().tolist()
                       )
def plot_2a(df):
    list_of_stuff=['ep-doublyrobust-nncme-dcme','ep-doublyrobust-dcme','ep-doublyrobustcorrect-dcme','ep-doublyrobustcorrect-nncme-dcme']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    plot_2_est_weights(dir=f'plot_2a',big_df=subset_df,
                       d_list=subset_df['D'].unique().tolist(),
                       methods=subset_df['mname'].unique().tolist(),
                       nlist=subset_df['n'].unique().tolist(),
                       data_list=subset_df['dataset'].unique().tolist()
                       )

def plot_3(df):

    if not os.path.exists('plot_3'):
        os.makedirs('plot_3')
    list_of_stuff =['ep-doublyrobust','ep-doublyrobust-nncme-dcme','ep-doublyrobust-dcme','ep-doublyrobust-nncme']+['doubleml','vanilla-dr','gformula','tmle','ipw','cf','bart','wmmd']
    mask = df['mname'].apply(lambda x: x in list_of_stuff ).values
    subset_df=df[mask]
    for ds in ['twins_2500','twins_2500_null']:
        for l in list_of_stuff:
            try:
                row = subset_df[(subset_df['mname']==l) &(subset_df['dataset']==ds)]
                ks_val=row['KS_pval'].tolist()[0]
                power = row['pval_005'].tolist()[0]

                big_df=pd.read_csv(row['final_res_path'].tolist()[0])
                sns.histplot(big_df,x='pval')
                if 'null' in ds:
                    plt.suptitle(f'KS test p-val: {ks_val}')
                else:
                    plt.suptitle(f'Power (level=0.05): {power}')
                plt.savefig(f'plot_3/{ds}_{l}.png',bbox_inches = 'tight')
                plt.clf()
            except Exception as e:
                print(e)
def plot_4(df): #Lalonde
    pass

def plot_5(df): #Inspire
    pass


def generate_latex(dir,filename_func,n_list=[500,5000],ds_names=[
         'unit_test',
          'distributions_middle_ground',
          'conditions_satisfied',
          'robin',
        'distributions',
        'distributions_uniform',
        'distributions_gamma',
    ]):

    n_cols = len(ds_names)
    n_rows = len(n_list)
    doc = Document(default_filepath=dir)
    col_width = 0.96/n_cols
    with doc.create(Figure(position='H')) as plot:
        with doc.create(subfigure(position='t', width=NoEscape(r'\linewidth'))):
            for i, n in enumerate(ds_names):
                if i == 0:
                    with doc.create(subfigure(position='H', width=NoEscape(r'0.04\linewidth'))):
                        # string_append = r'\raisebox{0cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{}}}' + '%'
                        string_append = '\hfill'
                        doc.append(string_append)
                with doc.create(subfigure(position='H', width=NoEscape(rf'{col_width}\linewidth'))):
                    name = f'{n}'
                    doc.append(Command('centering'))
                    doc.append(r'\rotatebox{0}{\scalebox{0.75}{%s}}' % name)
        counter = 0
        for idx, (i, j) in enumerate(itertools.product(n_list, ds_names)):
            if idx % (n_cols) == 0:
                name = f'$n={i}$'
                string_append = r'\raisebox{1.5cm}{\rotatebox[origin=c]{90}{\scalebox{0.75}{%s}}}' % name + '%\n'
            p = filename_func(dir,i,j) #f'{dir}/{j}_figure_5_{i}.png'
            string_append += r'\includegraphics[width=%f\linewidth]{%s}' % (col_width,p) + '%\n'
            counter += 1
            if counter == (n_cols):
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                counter = 0
    doc.generate_tex()

if __name__ == '__main__':
    """
    GPU - postprocessing
    """
    job_path='all_gpu_baselines_2'
    df_gpu = get_job_df(job_path)
    job_path='all_gpu_baselines_3'
    df_gpu_2 = get_job_df(job_path)
    job_path = 'all_cpu_baselines'
    df_cpu = get_job_df(job_path)
    df = pd.concat([df_gpu,df_gpu_2,df_cpu],axis=0).reset_index().drop(["index"], axis=1)
    df['b']=df['b'].apply(lambda x: float(x))
    new_df = df[df['n'].isin(['500','5000'])].reset_index().drop(["index"], axis=1)
    plot_1(new_df)
    plot_1a(new_df)
    plot_2(new_df)
    plot_2a(new_df)

    # job_path='all_gpu_real'
    # df_gpu = get_job_df_real(job_path)
    # job_path = 'all_cpu_real'
    # df_cpu = get_job_df_real(job_path)
    # df = pd.concat([df_gpu,df_cpu],axis=0).reset_index()
    # plot_3(df)
    generate_latex('plot_1',lambda dir,i,j:f'{dir}/{j}_figure_5_{i}.png')
    generate_latex('plot_1a',lambda dir,i,j:f'{dir}/{j}_figure_5_{i}.png')
    generate_latex('plot_2',lambda dir,i,j:f'{dir}/{j}_figure_5_{i}.png')
    generate_latex('plot_2a',lambda dir,i,j:f'{dir}/{j}_figure_5_{i}.png')
    list_of_stuff = ['ep-doublyrobust','ep-doublyrobust-nncme-dcme','ep-doublyrobust-dcme','ep-doublyrobust-nncme']+['doubleml','vanilla-dr','gformula','tmle','ipw','cf','bart','wmmd']
    generate_latex('plot_3',lambda dir,i,j:f'{dir}/{i}_{j}.png',['twins_2500','twins_2500_null'],list_of_stuff)






    #
    # job_path='all_cpu_baselines'
    # df = get_job_df(job_path)
    # df['b']=df['b'].apply(lambda x: float(x))
    # plot_2_est_weights(dir=f'{job_path}_plots',big_df=df,
    #                    d_list=df['D'].unique().tolist(),
    #                    methods=df['mname'].unique().tolist(),
    #                    nlist=df['n'].unique().tolist(),
    #                    data_list=df['dataset'].unique().tolist()
    #                    )

    # """
    # Plot 1
    # """
    # job_path='all_gpu_baselines_2'
    # df = get_job_df(job_path)
    # df['b']=df['b'].apply(lambda x: float(x))


