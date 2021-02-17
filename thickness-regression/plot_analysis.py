# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:20:40 2020

@author: mskim
"""
import matplotlib.pyplot as plt
from util import *

#%% layer 1
plt.figure()
start = 0
offset = 1
num = 5
inputs = range(10000,10010)
inputs = range(start,start+offset*num,offset)
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_n[i,:])
plt.legend(train_label[inputs,:],loc=0)
plt.show()

#%% layer 2
plt.figure()
start = 0
offset = 30
num = 5
inputs = range(10000,10010)
inputs = range(start,start+offset*num,offset)
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_n[i,:])
plt.legend(train_label[inputs,:],loc=0)
plt.show()

#%% layer 3
plt.figure()
start = 100000
offset = 900
num = 5
inputs = range(10000,10010)
inputs = range(start,start+offset*num,offset)
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_n[i,:])
plt.legend(train_label[inputs,:],loc=0)
plt.show()

#%% layer 4
plt.figure()
start = 0
offset = 27000
num = 5
inputs = range(10000,10010)
inputs = range(start,start+offset*num,offset)
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_n[i,:])
plt.legend(train_label[inputs,:],loc=0)
plt.show()

#%%
# start = 27000+900+30+1
start = 0
offset = 6
np.corrcoef([train_data[start+1*offset,:],train_data[start+30*offset,:],train_data[start+900*offset,:],train_data[start+27000*offset,:]])

[train_label[start+1*offset,:],train_label[start+30*offset,:],train_label[start+900*offset,:],train_label[start+27000*offset,:]]

inputs = range(start,start+offset,offset)
plt.figure()
plt.plot(train_data[start+1*offset,:])
plt.plot(train_data[start+30*offset,:])
plt.plot(train_data[start+900*offset,:])
plt.plot(train_data[start+27000*offset,:])
plt.plot(train_data[0,:],':')
plt.legend([train_label[start+1*offset,:],train_label[start+30*offset,:],train_label[start+900*offset,:],train_label[start+27000*offset,:],train_label[0,:]])
plt.show()

def find_signal(num_layer):
    tmp = (num_layer-10)/10
    tmp = tmp.astype(int)
    signal = 30**3*tmp[0] + 30**2*tmp[1] + 30**1*tmp[2] + 30**0*tmp[3]
    return signal


#%%
from util import *
result=[]
layer_2 = 200
layer_4 = 100
sums = 300
# train_n = norm_m(train_data,'minmax_f')
# train_n = mvgavg(train_data, 10, axis=1)

for j in range(1,10):
    layer_2 = 10*j
    layer_3 = 300-10*j
    for i in range(1,10):
        plt_1 = np.array([i*10,layer_2,sums-i*10,layer_4])
        plt_2 = np.array([sums-i*10,layer_2,i*10,layer_4])
        #plt.figure()
        plt.plot(train_n[find_signal(plt_1),:])
        
        #plt.plot(train_data[find_signal(plt_2),:])
        #plt.legend([plt_1,plt_2])
        # result.append(np.corrcoef(train_n[find_signal(plt_1),:],train_data[find_signal(plt_2),:]))
    plt.show()

for j in range(1,10):
    layer_2 = 10*j
    layer_3 = 300-10*j
    for i in range(1,10):
        plt_1 = np.array([i*10,layer_2,sums-i*10,layer_4])

        t = np.arange(207)
        sp = np.fft.fft(train_n[find_signal(plt_1),:])
        freq = np.fft.fftfreq(t.shape[-1])

        plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()
        #plt.plot(train_data[find_signal(plt_2),:])
        #plt.legend([plt_1,plt_2])
        # result.append(np.corrcoef(train_n[find_signal(plt_1),:],train_data[find_signal(plt_2),:]))

#%%
def find_signal(num_layer):
    tmp = (num_layer-10)/10
    tmp = tmp.astype(int)
    signal = 30**3*tmp[0] + 30**2*tmp[1] + 30**1*tmp[2] + 30**0*tmp[3]
    return signal

#%%
get_ipython().run_line_magic('matplotlib', 'inline')    
for i in range(40,60):
    plt.figure()
    plt.plot(train_n[find_signal(np.round(submit_2.values[i,1:]/10)*10),:],'--')
    plt.plot(train_n[find_signal(np.round(RF_r[i,:]/10)*10),:],'--')
    plt.plot(train_n[find_signal(RF_c[i,:]),:],'--')
    plt.plot(test[i,:],'r')
    plt.legend(['submit_2','RF_regressor','RF_classifier','True'])
    plt.show()


