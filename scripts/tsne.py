from __future__ import print_function
import numpy as np
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument(
    '-fw',
    '--first_word',
    type=str,
    required=True,
    help='Number of epochs to run the model',
    )

args = parser.parse_args()

first_word = args.first_word
folder_name = '../{}'.format(first_word)

feat_att = np.load(folder_name+'/feat_att.npy')
feat_vis = np.load(folder_name+'/feat_vis.npy')
feat_int = np.load(folder_name+'/feat_int.npy')
feat_grp = np.load(folder_name+'/feat_grp.npy')
labels = np.load(folder_name+'/labels.npy')

print(feat_att.shape, feat_grp.shape, feat_vis.shape, feat_int.shape, labels.shape)

X = feat_vis

feat_cols = ['ch_'+str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)

y = np.argmax(labels, axis=1)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

print( 'Size of the dataframe: {}'.format(df.shape) )

rndperm = np.random.permutation(df.shape[0])

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls", 29),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.show()
