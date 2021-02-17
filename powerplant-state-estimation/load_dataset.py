# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd 
import numpy as np
from multiprocessing import Pool 
import multiprocessing
from data_loader import data_loader #data_loader.py 파일을 다운 받아 주셔야 합니다. 
from tqdm import tqdm
from functools import partial
from pandas.api.types import is_string_dtype
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    CATlist = [] # categorical feature의 index 저장
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                # props[col].fillna(mn-1,inplace=True)
                props[col].fillna(0,inplace=True)
                CATlist.append(col)
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                CATlist.append(col)
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
        else: # for strings
            """
            # if on, convert to 1
            strs = props[col].values == 'ON'
            # strs = strs.values
            props.loc[strs , col] = 1
            props.loc[np.logical_not(strs) , col] = 0
            
            # else, all strings are filled with 0
            if np.sum(strs) == 0:
                props.loc[: , col] = 0
            
            CATlist.append(col)
            """
    # fill nan with 0
    # props.fillna(0)

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist, CATlist

def data_loader_all(func, path, train, start_row, nrows, **kwargs):
    '''
    Parameters:
    
    func: 하나의 csv파일을 읽는 함수 
    path: [str] train용 또는 test용 csv 파일들이 저장되어 있는 폴더 
    train: [boolean] train용 파일들 불러올 시 True, 아니면 False
    nrows: [int] csv 파일에서 불러올 상위 n개의 row 
    lookup_table: [pd.DataFrame] train_label.csv 파일을 저장한 변수 
    event_time: [int] 상태_B 발생 시간 
    normal: [int] 상태_A의 라벨

    Return:
    
    combined_df: 병합된 train 또는 test data
    '''
    
    # 읽어올 파일들만 경로 저장 해놓기 
    files_in_dir = os.listdir(path)
    
    files_path = [path+'/'+file for file in files_in_dir]
    if train :
        func_fixed = partial(func, start_row=start_row, nrows = nrows, train = True, lookup_table = kwargs['lookup_table'], event_time = kwargs['event_time'], normal = kwargs['normal'])
    else : 
        func_fixed = partial(func, start_row=start_row,nrows = nrows, train = False)
        # 아래 문장의 의미: __name__이 __main__일 때 실행되는 것이므로, 
        # load_dataset을 import한 다른 module에서는 절대 실행되지 않고 이 파일을 직접 실행시킨 경우에만 실행됨
    # if __name__ == '__main__':
    if train:
        df_list = []
        label_list = []
        # cat = np.ones((1,5123),dtype=bool)
        for i in range(len(files_path)):
            tmp_data, tmp_label = func_fixed(path = files_path[i])
            # tmp_data = func_fixed(path = files_path[i])
            df_list.append(tmp_data)
            label_list.append(tmp_label)
            print(i,'th iteration')
            # cat = cat * tmp_catlist
        data = pd.concat(df_list, ignore_index=True,sort=False)
        data, NAlist, CATlist = reduce_mem_usage(data)
        print("_________________")
        print("")
        print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
        print("_________________")
        print("")
        print(NAlist)
        CATlist.append('id')
        # catlist를 원하는 형식으로 바꿈
        """
        bool_list = data.loc[0]
        bool_list[:] = False
        bool_list[CATlist] = True
        bool_list = bool_list.values
        """
        combined_label = pd.concat(label_list, ignore_index=True,sort=False)
        return data, combined_label, CATlist
    else:
        df_list = []
        for i in range(len(files_path)):
            tmp_data= func_fixed(path = files_path[i])
            df_list.append(tmp_data)

            print(i,'th iteration')
        data = pd.concat(df_list, ignore_index=True,sort=False)
        data, NAlist, CATlist = reduce_mem_usage(data)
        return data



train_path = 'data_raw/train'
test_path = 'data_raw/test'
label = pd.read_csv('data_raw/train_label.csv')
# train load
train, train_label, _ = data_loader_all(data_loader, path = train_path, train = True, start_row = 10,nrows = 50, normal = 999, event_time = 0, lookup_table = label)
# test load
# test= data_loader_all(data_loader, path = test_path, train = False,start_row = 20,nrows = 2)


"""
tmp = train.V0020
real_train_i = []
for i in range(len(train)):
    if isinstance(tmp[i], str):
        real_train_i.append(i)

tmp = test.V0020
real_test_i = []
for i in range(len(test)):
    if isinstance(tmp[i], str):
        real_test_i.append(i)
"""
