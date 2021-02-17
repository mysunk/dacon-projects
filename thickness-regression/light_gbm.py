# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:30:06 2020

@author: mskim
"""
import scipy
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

####################################################train-test split
tmp = np.sum(train_label,axis=1)
y = tmp/10
y = y.astype(int)-4
y = np.reshape(y,(810000,))
X_train, X_test, y_train, y_test = train_test_split(train_n, y, test_size=0.2, random_state=42)

param = {'max_depth':30 , 'objective': 'multiclass',"num_class" : 117}
# param = {'num_leaves': 30, 'objective': 'regression','metric':'l1'}
# num_round = 10

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

bst = lgb.train(param, train_data, valid_sets=test_data,num_boost_round=100)

y_pred = bst.predict(X_test)
y_pred_max = np.argmax(y_pred,axis=1)

####################################################cv
tmp = np.sum(train_label,axis=1)
y = tmp/10
y = y.astype(int)-4
y = np.reshape(y,(810000,))

# param = {'num_leaves': 30, 'objective': 'multiclass',"num_class" : 117 }
param = {'num_leaves': 20, 'objective': 'multiclassova',"num_class" : 117 }
# param = {'num_leaves': 30, 'objective': 'regression','metric':'l1'}
# num_round = 10

train_data = lgb.Dataset(train_n, label=y)
bst = lgb.cv(param, train_data, num_boost_round=10,nfold=4,stratified=True,verbose_eval=True)

y_pred = bst.predict(X_test)
y_pred_max = np.argmax(y_pred,axis=1)