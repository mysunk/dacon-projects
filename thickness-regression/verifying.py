# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:13:53 2020

@author: mskim
"""
import pandas as pd
import numpy as np
from mvgavg import mvgavg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 
train_data = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/data/train.csv', engine='python')
test_data = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/data/test.csv', engine='python')

train_raw = train_data.values[:,4:]
train_raw = train_raw.astype(float)
train_label = train_data.values[:,0:4]
train_label = train_label.astype(int)
test_raw = test_data.values[:,1:]
test_raw = test_raw.astype(float)

#%% # standard scaling

scaler = StandardScaler()
scaler.fit(train_raw)
train= scaler.transform(train_raw)
test = scaler.transform(test_raw)

#%% # moving average

train_n = mvgavg(train,5, axis=1)
test_n = mvgavg(test, 5, axis=1)

#%% Save
np.save('./data/train_n.npy',train_n)
np.save('./data/test_n.npy',test_n)
np.save('./data/train_raw.npy',train_raw)
np.save('./data/test_raw.npy',test_raw)

#%% plot
plt.figure
plt.plot(train_raw[1100,:])
plt.plot(train[1100,:])
plt.plot(train_n[1100,:])
plt.legend(['raw','scaled','moving averaged'])
plt.show