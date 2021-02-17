import pandas as pd
import numpy as np
from mvgavg import mvgavg

#%% train dataset

path = 'D:/Users/msun/Desktop/ISP/Daycon'
train = pd.read_csv(path + '/train.csv', engine='python')
train_data = train.values[:,4:]
train_data = train_data.astype(float)
train_label = train.values[:,0:4]
train_label = train_label.astype(int)

test = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/data/test.csv', engine='python')
sample = pd.read_csv(path + '/data/sample_submission.csv', engine='python')
test = test.values[:,1:]
test = test.astype(float)
#%%
train_data = np.load('./data/train_raw.npy')
train_label = np.load('./data/train_label.npy')



#%%
"""
layer1= np.unique(train_label[:,0])
layer2 = np.unique(train_label[:,1])
layer3 = np.unique(train_label[:,2])
layer4 = np.unique(train_label[:,3])
"""
num_layer = np.linspace(10,300,30)

#%%
result_1 = np.load('./result_maxdepth_3to7_layer_1.npy')
result_2 = np.load('./result_maxdepth_8to12_layer_1.npy')

#%%
test = np.load('./data/test.npy')
train_label = np.load('./data/train_label.npy')


#%%
train_n = mvgavg(train_data, 20, axis=1)
test = mvgavg(test, 20, axis=1)

#%%
# value = np.load('D:/Users/msun/Desktop/ISP/00. 서버컴퓨터용/200107/submission_2.npy')
sample = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/data/sample_submission.csv', engine='python')
sample['layer_1'] = RF_c[:,0]
sample['layer_2'] = RF_c[:,1]
sample['layer_3'] = RF_c[:,2]
sample['layer_4'] = RF_c[:,3]
sample.to_csv('submit_6.csv',index=False)

#%%
submit_2 = pd.read_csv('D:/Users/msun/Desktop/ISP/Daycon/result/submit_2.csv', engine='python')
RF_c = np.load('D:/Users/msun/Desktop/ISP/Daycon/result/RF_classifier.npy')
RF_r = np.load('D:/Users/msun/Desktop/ISP/Daycon/result/RF_regressor.npy')
