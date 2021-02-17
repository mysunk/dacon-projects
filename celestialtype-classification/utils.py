import pandas as pd
import glob, os
import csv
try:
    import cPickle as pickle
except BaseException:
    import pickle


def to_number(x, dic):
    return dic[x]  

def load_dataset(n_stack): # stacking에 따라 다른 dataset load -- not implemented
    # load dataset
    train = pd.read_csv('data_raw/train.csv', index_col=0)
    sample_submission = pd.read_csv('data_raw/sample_submission.csv', index_col=0)

    # make label
    column_number = {}
    for i, column in enumerate(sample_submission.columns):
        column_number[column] = i

    train['type_num'] = train['type'].apply(lambda x: to_number(x, column_number))
    train['fiberID'] = train['fiberID'].astype(int)

    train_label = train['type_num']
    train = train.drop(columns=['type', 'type_num'], axis=1)
    return train, train_label
    
def save_obj(obj, name):
    try:
        with open('trials/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('trials')           

def load_obj(name):
    with open('trials/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def change_param(cv_result):
    cv_best_param = cv_result.best_trial['result']['param']['cv_params']
    nfold = cv_best_param['nfold']
    del cv_best_param['nfold']
    tree_best_param = cv_result.best_trial['result']['param']['tree_params']
    tree_best_param['n_estimators'] = cv_best_param.pop('num_boost_round') # change name
    tree_best_param['seed'] = cv_best_param.pop('seed')
    cv_best_param['eval_metric'] = tree_best_param.pop('eval_metric')
    param = {'fit_params':cv_best_param, 'tree_params':tree_best_param}
    return param, nfold