# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:35:53 2020

@author: mskim
"""

import lightgbm as lgb
# import xgboost as xgb
# import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
import numpy as np

class Params(object):
    
    def __init__(self, cv_params):
        self.param = dict()
        self.param['cv_params'] = cv_params
        
    def xgb_params(self):
        # XGB parameters
        xgb_tree_params = {
            'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
            'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample':        hp.uniform('subsample', 0.8, 1),
            'verbose':                  -1,
            'num_class':                19,
            'n_jobs':                   -1,
            'objective':                'multiclass',
            'eval_metric':              'mlogloss',
        }
        self.param['tree_params'] = xgb_tree_params
    
    def lgb_params(self):
        # LightGBM parameters
        lgb_tree_params = {
            # for tuned
            'learning_rate':            hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
            'max_depth':                hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
            'min_child_weight':         hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'colsample_bytree':         hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample':                hp.uniform('subsample', 0.8, 1),
            'verbose':                  -1,
            'num_class':                19,
            'n_jobs':                   -1,
            'objective':                'multiclass',
            'eval_metric':              'multi_logloss',
        }
        self.param['tree_params'] = lgb_tree_params
    
    def ctb_params(self):
        # CatBoost parameters
        ctb_tree_params = {
            'learning_rate':     hp.choice('learning_rate',     np.arange(0.05, 0.31, 0.05)),
            'max_depth':         hp.choice('max_depth',         np.arange(5, 16, 1, dtype=int)),
            'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'verbose':                  -1,
            'num_class':                19,
            'n_jobs':                   -1,
            'objective':                'multiclass',
            'eval_metric':              'mlogloss',
        }
        self.param['tree_params'] = ctb_tree_params
    
    def get_param(self):
        return self.param

class HPOpt_cv(object):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
    
    def xgb_cv(self, param):
        # Not implemented
        pass

    def lgb_cv(self, param):
        train_set = lgb.Dataset(self.x_train, self.y_train)
        cv_results = lgb.cv(param['tree_params'], train_set, **param['cv_params'])
        loss = min(cv_results[param['tree_params']['eval_metric'] + '-mean'])
        return {'loss': loss, 'param':param, 'status': STATUS_OK}
    
    def ctb_cv(self, param):
        # Not implemented
        pass