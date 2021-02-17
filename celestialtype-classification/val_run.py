# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:15:14 2020

@author: mskim
"""
from utils import *
from main_objs import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='GBDT validation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf_filename", default='tmp', type=str)
    args= vars(parser.parse_args())
    
    # sort params in loss ascending order --- not implemented
    param, nfold = change_param(load_obj(args['conf_filename']))
    
    # load dataset
    train, train_label = load_dataset(1) # stack 번호
    
    # train-test split
    skf = StratifiedKFold(n_splits=nfold, random_state=None, shuffle=False)
    
    oof_loss = np.zeros((nfold,2))
    oof_pred = np.zeros((train.shape[0],param['tree_params']['num_class']))
    
    for i, (train_index, test_index) in enumerate(skf.split(train.values, train_label.values)):
        x_train = train.iloc[train_index]
        y_train = train_label.iloc[train_index]
        x_test = train.iloc[test_index]
        y_test = train_label.iloc[test_index]
        
        # make classifier
        obj = Gbdt(x_train, x_test, y_train, y_test, param)
        loss = obj.lgb_clf()
        oof_loss[i] = (loss['train_loss'],loss['val_loss'])
        
        # oof prediction
        oof_pred[test_index,:] = obj.predict_clf(x_test)
        break
    
    # save result -- not implemented