from cv2 import kmeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

desc_file = 'csv/idps_desc.csv'


def get_quantile(df, q_num=3):
    d = df.copy()
    d['Q'] = pd.qcut(d['mean'], q_num, labels=False) + 1
    return d


def get_quantile_kmeans(df, n_centroid=4):
    d = df.copy()
    kmeans = KMeans(n_clusters=n_centroid, random_state=0).fit(d['mean'].to_numpy().reshape(-1,1))
    d['Q'] = kmeans.labels_.astype(int)
    
    #print(kmeans.cluster_centers_)
    return d


def get_grouped_desc_2(file):
    desc = pd.read_csv(file, sep=';', index_col='id', skipinitialspace=True)
    desc.head()
    
    volume = desc[desc['desc'].str.match('Volume')]
    area = desc[desc['desc'].str.match('Area')]
    grey_white = desc[desc['desc'].str.match('Grey-white')]
    mean_thickness = desc[desc['desc'].str.match('Mean thickness')]
    mean_intensity = desc[desc['desc'].str.match('Mean intensity')]

    return {
        'volume':volume, 
        'area':area, 
        'grey white':grey_white, 
        'mean thickness':mean_thickness,
        'mean intensity': mean_intensity}


def get_grouped_desc(desc_file, multi_r2, multi_mse, name='result', write_result=False):
    desc = pd.read_csv(desc_file, index_col='id')
    r2 = pd.read_csv(multi_r2).iloc[[-1]].T # select the last entry
    mse = pd.read_csv(multi_mse).iloc[[-1]].T

    mse.columns =['mse']
    mse.index.name = 'id'
    mse.index = mse.index.astype(int)

    r2.columns=['r2']
    r2.index.name = 'id'
    r2.index = r2.index.astype(int)

    joined = r2.join(mse, on='id', how='left')
    desc = desc.join(joined, on='id', how='left')

    if write_result:
        desc.to_csv('results/'+name+'.csv', sep=';',doublequote=False, escapechar='\t')

    volume = desc[desc['desc'].str.match('Volume')]
    area = desc[desc['desc'].str.match('Area')]
    grey_white = desc[desc['desc'].str.match('Grey-white')]
    mean_thickness = desc[desc['desc'].str.match('Mean thickness')]
    mean_intensity = desc[desc['desc'].str.match('Mean intensity')]

    # print('total : {}'.format(volume.shape[0] + area.shape[0] + grey_white.shape[0] + mean_thickness.shape[0] + mean_intensity.shape[0]))
    return {
        'volume':volume, 
        'area':area, 
        'grey white':grey_white, 
        'mean thickness':mean_thickness,
        'mean intensity': mean_intensity}


def boxplot(result_list, names, q_num=3, group='volume', title='', use_kmeans=False, ax=None, legend=False):
    if len(title) == 0:
        title = group

    q = list()
    for i in range(len(result_list)):
        if use_kmeans:
            df = get_quantile_kmeans(result_list[i][group], n_centroid=q_num)
        else:
            df = get_quantile(result_list[i][group], q_num=q_num)
        df['model'] = names[i]
        q.append(df[['r2', 'mse', 'Q', 'model']].reset_index(drop=True))

    ret = q[0]
    for i in range(1, len(q)):
        ret = pd.concat([ret, q[i]], axis=0)
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1)
        fig.set_size_inches(10,5)    

    #sns.boxplot(data=ret, x='Q', y='r2', hue='model', ax=ax[0])
    sns.boxplot(data=ret, x='Q', y='r2', hue='model', ax=ax)
    #sns.boxplot(data=ret, x='Q', y='mse', hue='model', ax=ax[1])   
    ax.set_xlabel('Quantile')
    ax.set_ylabel('R2 Score')
    ax.set_title(title)

    if not legend:
        ax.get_legend().remove()

    return ret


def process_results(filename, index=0):
    histfile = filename + '_' + str(index) + '_hist.csv'
    msefile = filename + '_test_multi_mse.csv'
    r2file = filename + '_test_multi_r2.csv'

    hist_pd = pd.read_csv(histfile)
    grouped_desc = get_grouped_desc(desc_file, r2file, msefile)

    return hist_pd, grouped_desc


def plot_time_series(history_list, labels, group='loss', title=None, log_y=False, ax=None, legend=False):
    if title==None:
        title = group

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10,5)
    
    for i in range(len(history_list)):
        history_list[i][group].plot(ax=ax, label=labels[i])
        
    ax.set_xlabel('epochs')
    ax.set_ylabel(group)
    ax.grid(True)
    ax.set_title(title)

    if legend:
        ax.legend(title='model')

    if log_y:
        ax.set_yscale('log')

    