# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:57:36 2021

@author: shossein
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#%%
path = r'C:\Users\shossein\Desktop'
df = pd.read_csv(path + '\Book1.csv', index_col=0)
print(df.head())
#%%
train= df.iloc[:,0:5].values
train
train.reshape(1,-1)
#%%
target = df.index
print(target)
#%%
number=LabelEncoder()
target=number.fit_transform(target)
print(target)
#%%
scaler=StandardScaler()
data_x = scaler.fit_transform(train)
print(data_x)
#%%
model=TSNE(n_components=3, random_state=0)
tsne_data=model.fit_transform(data_x)
print(tsne_data)
#%%
plt.scatter(tsne_data[:,0], tsne_data[:,1], c=target )


