# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:38:32 2020

@author: mskim
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import *
from scipy.stats import linregress

#%% USER
want_plot = train_data
offset = 1

#%%
for i in range(30):
    plt.figure()
    for j in range(5):
        plt.plot(want_plot[i+offset*j,:]) # 1~3 고정(10), 4만 10~300
    plt.legend(train_label[range(i,i+offset*(j+1),offset),:])
    plt.show()
    
#%% plot residual
plt.figure()
for i in range(3):
    # 4만 변했음 (10 to 50)
    plt.plot(-want_plot[0,:] + want_plot[i+offset,:])
plt.show()
    
plt.figure()
for i in [59]:
    # 4만 변했음 (10 to 50)
    plt.plot(-want_plot[i,:] + want_plot[i+offset,:])
plt.show()

### 59번째, 29번째가 다르게 생긴듯?
### 29번째는 30에서 29를 뺀 것, 59번째는 60에서 59를 뺀 것 -- layer 3의 문제라기보단 4가 300에서 10 빼서 그런 듯

#%% 한 개씩
# train_n = norm_m(train_data,'minmax')
plt.figure()
offset=27000 # 1 30 900 27000
start = 10000
num_plot = 3
inputs = range(start,start+offset*3,offset) 
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_n[i,:])
    plt.legend(train_label[inputs,:])
    plt.show()

#%% 한 개씩
plt.figure()
start = 0+30*5
offset = 30
num = 5
inputs = range(10000,10010)
inputs = range(start,start+offset*num,offset)
for i in inputs:
    # 4만 변했음 (10 to 50)
    plt.plot(train_data[i,:])
plt.legend(train_label[inputs,:],loc=0)
plt.show()


#%% pca
pca = PCA(n_components=100)
pca.fit(train_data)

#%%
plt.figure()
plt.plot(np.transpose(pca.components_[0:3,:]))
plt.legend('1','2')
plt.show()
#%%
pca = PCA(n_components=10, svd_solver='full')
pca.fit(train_1)

#%% find slope
from util import *
# train_n = np.transpose(train_n)
x_axis = np.linspace(0,22.6,226)
tmp = train_n[27000*2]
# (np.mean(x_axis*tmp,axis=1) - np.mean(x_axis)*np.mean(tmp,axis=1)) / (np.mean(x_axis**2) - np.mean(x_axis)**2)

tmp_slope = np.gradient(tmp,x_axis)


plt.figure()
plt.plot(x_axis,tmp)
plt.plot(x_axis,tmp_slope)
plt.show()

#%%
y = [2,6,12]
np.gradient(y,x)
#%%
from sklearn.multioutput import MultiOutputRegressor

#%%
for i in range(200,300,1):
    plt.plot(test[i,:])
    plt.show()
    time.sleep(0.1)
    
#%%

t = np.arange(207)
sp = np.fft.fft(train_n[10000,:])
freq = np.fft.fftfreq(t.shape[-1])

plt.plot(freq, sp.real, freq, sp.imag)
plt.show()

