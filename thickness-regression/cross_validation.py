# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:42:06 2020

@author: mskim
"""

#%% cross validation partition #########################################################################
skf = StratifiedKFold(n_splits=10,random_state=0)
# for layer 1
label = train_label[:,0]
results = np.zeros((skf.n_splits,len(inputs)))

i=0
for train_index, test_index in skf.split(train_n, label):
    results[i,:] = Parallel(n_jobs=6)(delayed(cv_form)(j) for j in inputs) # j is max depth
    i=i+1
    print('=======executing======')
"""
def cv_form(j):
    return fitting(train_n, train_index, test_index, label, j)

def fitting(data, train_index, test_index, label, j):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    clf = RandomForestClassifier(max_depth=j, random_state=0,n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result = np.mean(y_pred == y_test)
    return result
"""