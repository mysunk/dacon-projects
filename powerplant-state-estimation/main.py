# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:48:49 2020

@author: mskim
"""

from load_dataset import *
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

#%%=============================================================================
# load dataset -- train
start_row = 0
nrows = 60

train_path = 'data_raw/train'
label = pd.read_csv('data_raw/train_label.csv')
train, train_label, cat_list = data_loader_all(data_loader, path = train_path, train = True, start_row = start_row,nrows = nrows, normal = 999, event_time = 10, lookup_table = label)


# train = pd.read_csv('data_npy/train_agg.csv')
# train_label = pd.read_csv('data_npy/train_label_agg.csv')
"""
cat_list = []
# open file and read the content in a list
with open('data_npy/cat_list.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        cat_list.append(currentPlace)
        
        
column_name = train.columns
column_name = list(column_name)
# train = train.values
train_id = train_label.id.values
train_label = train_label.label.values


res = set(column_name).difference(cat_list)
train = train[res]
train = train.values
"""
(train = train.values
train_label = train_label.label)

#%%=============================================================================
# Initialize a model with small rounds
param = { 'num_leaves': 500,
         'num_class':198,
         'objective': 'multiclass',
         'metric':'multi_logloss',
         'learning_rate':0.1,
         'feature_fraction':0.01,
         # 'bagging_freq':5,
         # 'importance_type':'gain',
           'bagging_fraction' : 1,
           # 'min_split_gain':0.005,
         'verbose':2,
         'tree_learner':'feature',
           'n_jobs':-1}

# train and val split
skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)

train_i = []
test_i = []
lgb_train = []
lgb_val = []
for train_index, test_index in skf.split(train, train_id):
    train_i.append(train_index)
    test_i.append(test_index)
    lgb_train.append(lgb.Dataset(train[train_index,:], label=train_label[train_index],free_raw_data=False))
    lgb_val.append(lgb.Dataset(train[test_index,:], label=train_label[test_index],free_raw_data=False))
    

models = []

for i in range(4):
    train_index = train_i[i]
    test_index = test_i[i]
    # Make lgb dataset
    # lgb_train = lgb.Dataset(X, label=y,free_raw_data=False,feature_name=column_name, categorical_feature=cat_list)
    # lgb_val = lgb.Dataset(X_val, label=y_val,free_raw_data=False,feature_name=column_name, categorical_feature=cat_list)
    # lgb_train = lgb.Dataset(train[train_index,:], label=train_label[train_index],free_raw_data=False)
    # lgb_val = lgb.Dataset(train[test_index,:], label=train_label[test_index],free_raw_data=False)
    
    gbm = lgb.train(param, lgb_train[i],num_boost_round=10,valid_sets=[lgb_train[i], lgb_val[i]],early_stopping_rounds=5)

    print('[split'+str(i)+'] Finished initial training ...')
    model_name = 'model_lgb/'+'model'+str(i)+'.txt'
    gbm.save_model(model_name)
    models.append(gbm)
    
#%%=============================================================================
# With initial model -- basic
for i in range(4):
    train_index = train_i[i]
    test_index = test_i[i]
    
    # Make lgb dataset
    # lgb_train = lgb.Dataset(train[train_index,:], label=train_label[train_index],free_raw_data=False)
    # lgb_val = lgb.Dataset(train[test_index,:], label=train_label[test_index],free_raw_data=False)
    
    # model_name = 'model_lgb/'+'model'+str(i)+'.txt'
    models[i] = lgb.train(param, lgb_train[i], num_boost_round=100,valid_sets=[lgb_train[i], lgb_val[i]], init_model=models[i],early_stopping_rounds=3)
    
    print('[split'+str(i)+'] Finished 2nd training ...')
    # gbm.save_model(model_name)

#%%=============================================================================
# Varying parameter -- learning rate
for i in range(4):
    train_index = train_i[i]
    test_index = test_i[i]
    
    # Make lgb dataset
    # lgb_train = lgb.Dataset(train[train_index,:], label=train_label[train_index],free_raw_data=False)
    # lgb_val = lgb.Dataset(train[test_index,:], label=train_label[test_index],free_raw_data=False)
    # model_name = 'model_lgb/'+'model'+str(i)+'.txt'
    n_boost_round = 10
    #val_for_each = int(n_boost_round/5)
    models[i] = lgb.train(param, lgb_train[i], num_boost_round=n_boost_round,valid_sets=[lgb_train[i], lgb_val[i]], init_model=models[i],
                    learning_rates=[0.05]*n_boost_round,
                    early_stopping_rounds=3)
    
    print('[split'+str(i)+'] Finished 3rd training ...')
    # gbm.save_model(model_name)
    
#%%=============================================================================
# Varying parameter -- other parameters
for i in range(4):
    train_index = train_i[i]
    test_index = test_i[i]
    
    # Make lgb dataset
    # lgb_train = lgb.Dataset(train[train_index,:], label=train_label[train_index],free_raw_data=False)
    # lgb_val = lgb.Dataset(train[test_index,:], label=train_label[test_index],free_raw_data=False)
    
    # model_name = 'model_lgb/'+'model'+str(i)+'.txt'
    n_boost_round = 100
    val_for_each = int(n_boost_round/5)
    models[i] = lgb.train(param, lgb_train[i], num_boost_round=n_boost_round,valid_sets=[lgb_train[i], lgb_val[i]], init_model=models[i],
                    callbacks=[lgb.reset_parameter(feature_fraction=[0.001]*100)],
                    early_stopping_rounds=10)
    
    print('[split'+str(i)+'] Finished 3rd training ...')
    # gbm.save_model(model_name)

#%%======================save model
for i in range(4):
    model_name = 'model_lgb/'+'model'+str(i)+'.txt'
    models[i].save_model(model_name)