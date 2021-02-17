# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:40:43 2020

@author: mskim
"""
from hyperopt import hp
import numpy as np
from tuning_with_cv import * # paramter settings # objects for parameter tuning
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from utils import *
import argparse

if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='GBDT general fit configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_boost_round", default=10, type=int)
    parser.add_argument("--early_stopping_rounds", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nfold", default=10, type=int)
    cv_params = vars(parser.parse_args())
    
    parser_2 = argparse.ArgumentParser(description='Others',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_2.add_argument("--max_evals", default=2, type=int)
    parser_2.add_argument("--filename", default='tmp', type=str)
    parser_2.add_argument("--n_stack", default=1, type=int)
    args = parser_2.parse_args()
    
    # load and combine parameters
    params = Params(cv_params = cv_params)
    params.lgb_params()
    
    # load dataset
    train, train_label = load_dataset(args.n_stack)

    bayes_trials_1 = Trials()
    obj = HPOpt_cv(train,train_label)
    lgb_opt = obj.process(fn_name='lgb_cv', space=params.get_param(), trials=bayes_trials_1, algo=tpe.suggest, max_evals=args.max_evals)
    
    # save trial
    save_obj(bayes_trials_1,args.filename)
    # tmp = load_obj('tmp')