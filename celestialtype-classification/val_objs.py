# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:57:50 2020

@author: mskim
"""

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class Gbdt(object):

    def __init__(self, x_train, x_test, y_train, y_test, param):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.fit_params = param['fit_params']
        self.tree_params = param['tree_params']
        self.clf = None # not assigned yet
        
    def xgb_clf(self, param):
        clf = xgb.XGBClassifier(**self.tree_params)
        return self.train_clf(clf)

    def lgb_clf(self):
        self.clf = lgb.LGBMClassifier(**self.tree_params)
        return self.train_clf()

    def ctb_clf(self, param):
        clf = ctb.CatBoostClassifier(**self.tree_params)
        return self.train_clf(clf)

    def train_clf(self, param=None):
        """
        # for init model
        lightgbm 2.3.2가 API에 업데이트 돼야 사용할 수 있음..
        아마 예제에서 이걸 쓴 거는 단순히 validation 하려고 인듯.. 학습에는 적절하지 않아보임
        다른 모델은 사용 가능한지 확인 안해봄
        """
    
        # train
        self.clf.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **self.fit_params)
        
        # save result of train, val
        fit_result = {
            'train_loss': self.clf.best_score_['training'][self.fit_params['eval_metric']],
            'val_loss':self.clf.best_score_['valid_1'][self.fit_params['eval_metric']] }
        
        return fit_result

    def predict_clf(self, test):
        # prediction result of validation set
        pred = self.clf.predict_proba(test)
        return pred
