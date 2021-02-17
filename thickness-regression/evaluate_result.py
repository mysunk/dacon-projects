# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:36:52 2020

@author: mskim
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

#%% load dataset
# load result
result = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/submit/lgb_submit_1.csv', engine='python')
result = result.values[:,1:]
result = result.astype(float)

result1 = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/submit/lgb_submit_10.csv', engine='python')
result1 = result1.values[:,1:]
result1 = result1.astype(float)

result2 = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/submit/lgb_submit_7.csv', engine='python')
result2 = result2.values[:,1:]
result2 = result2.astype(float)

# load original training set
train_raw = np.load('./data/train_raw.npy')
test_raw = np.load('./data/test_raw.npy')

# save
sample = pd.read_csv('./data/sample_submission.csv')
sample.layer_1 = result1[:,0]
sample.layer_2 = result1[:,1]
sample.layer_3 = result1[:,2]
sample.layer_4 = result1[:,3]
sample.to_csv('./submit/lgb_submit_new.csv',index = False)

#%% plot
def find_signal(num_layer):
    tmp = (num_layer-10)/10
    tmp = tmp.astype(int)
    signal = 30**3*tmp[0] + 30**2*tmp[1] + 30**1*tmp[2] + 30**0*tmp[3]
    return signal


for i in range(100,110):
    plt.figure()
    plt.plot(train_raw[find_signal(np.round(result1[i,:]/10)*10),:],'--')
    plt.plot(train_raw[find_signal(np.ceil(result1[i,:]/10)*10),:],'--')
    plt.plot(train_raw[find_signal(np.floor(result1[i,:]/10)*10),:],'--')
    # plt.plot(train_raw[find_signal(np.round(result2[i,:]/10)*10),:],'--')
    plt.plot(test_raw[i,:],'r')
    # plt.legend(['result 1','result 2','True'])
    plt.legend(['round','ceil','floor','True'])
    # plt.legend(['round','True'])
    plt.show()
    




