# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:08:17 2021

@author: shossein
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from os import chdir, getcwd
import seaborn as sns
#%%
path = r'C:\Users\shossein\Desktop'
df = pd.read_csv(path + '\Book1.csv', index_col=0)
print(df.head())
#%%
print(df.shape)
scaled_data= preprocessing.scale(df)
#%%
pca =SparsePCA(n_components=6, random_state=123)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
# per_var =np.round(pca.explained_variance_ratio_* 100, decimals=1)
# labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]
# plt.bar(x=range (1, len(per_var)+1), height= per_var, tick_label=labels)
# plt.show()
#%%
x = df # Load boston dataset
 # select non-categorical variables
scaled_data= preprocessing.scale(x)
spca = SparsePCA(alpha=1e-5, ridge_alpha=1e-6, normalize_components=True) # Solve sparse pca
spca.fit(scaled_data)
t_spca = spca.transform(scaled_data)
p_spca = spca.components_
#%%
# Calculate the explained variance as explained in sparse pca original paper
t_spca_qr = np.linalg.qr(t_spca) # QR decomposition of modified PCs
q = t_spca_qr[0]
r = t_spca_qr[1]
# compute adjusted variance
variance = []
for i in range(len(r)):
    variance.append(np.square(r[i][i]))
variance = np.array(variance)/np.size(scaled_data)
# compute variance_ratio
total_variance_in_x = np.matrix.trace(np.cov(scaled_data)) # Variance in the original dataset
explained_variance_ratio = np.cumsum(variance / total_variance_in_x)
#%%
per_var =np.round(explained_variance_ratio * 100, decimals=1)
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]
plt.bar(x=range (1, len(per_var)+1), height= per_var, tick_label=labels)
plt.show()
#%%

pca_df = pd.DataFrame(t_spca, index =df.index, columns=labels)
sns.scatterplot(pca_df.PC3, pca_df.PC4, hue=pca_df.index.tolist())
# for sample in pca_df.index[0:1]:
#     try:
#         plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC4.loc[sample]))
#     except:
#         pass