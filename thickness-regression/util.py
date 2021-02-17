# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:33:26 2020

@author: mskim
"""
# for normalize
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from numpy import *

def norm_m(X, types):
    if types == 'raw':
        return X
    if types == 'rms': 
        return normalize(X,norm='l2',axis=1)
    if types == 'minmax':
        scaler= MinMaxScaler(feature_range=(0,1))
        return transpose(scaler.fit_transform(transpose(X)))
    if types == 'minmax_f':
        scaler= MinMaxScaler(feature_range=(0,1))
        return scaler.fit_transform(X)
    
    
#def onehot_encode():
def score(actual, predicted):
    return mean_absolute_error(actual, predicted)
    