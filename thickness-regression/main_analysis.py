# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:08:49 2020

@author: mskim
"""
import pandas as pd
import numpy as np
from mvgavg import mvgavg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#%% load train and test
train_n = np.load('./data/train_n.npy')
test = np.load('./data/test.npy')
train_label = np.load('./data/train_label.npy')
submit_cadidate = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/submit/lgb_submit_2.csv', engine='python')

scaler = StandardScaler()
scaler.fit(train_n)
train_n= scaler.transform(train_n)
test = scaler.transform(test)

#%% Make figure inline
# For only first step
get_ipython().run_line_magic('matplotlib', 'inline')    

#%%
def find_signal(num_layer):
    tmp = (num_layer-10)/10
    tmp = tmp.astype(int)
    signal = 30**3*tmp[0] + 30**2*tmp[1] + 30**1*tmp[2] + 30**0*tmp[3]
    return signal


for i in range(10):
    plt.figure()
    plt.plot(train_n[find_signal(np.round(submit_cadidate.values[i,1:]/10)*10),:],'--')
    plt.plot(test[i,:],'r')
    plt.legend(['submit_candidate','True'])
    plt.show()
    