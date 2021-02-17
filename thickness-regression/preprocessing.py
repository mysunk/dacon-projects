import numpy as np
# for training
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# for parallel loop
from joblib import Parallel, delayed
import multiprocessing
from util import *
from sklearn.model_selection import GridSearchCV
from scipy.fftpack import fft

# for multi-threading
def cv_form(j):
    print('===========executing==============')
    return fitting(X_train, X_test, y_train, y_test,j)

def fitting(X_train, X_test, y_train, y_test, j):
    clf = RandomForestClassifier(max_depth=j, random_state=0,n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result = score(y_test, y_pred) # MAE
    return result


#%% user#########################################################################
norm_type = 'rms'
inputs = range(15,40,5)
num_layer = 0 # 1st layer

#%% normalize#########################################################################
# train_n = norm_m(train_data,norm_type)

#%% Validation#########################################################################
X = train_data
y = train_label[:,num_layer]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)
# results = Parallel(n_jobs=5)(delayed(cv_form)(j) for j in inputs) # j is max depth


#%% training and testing#########################################################################
clf = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
result = score(y_test, y_pred) # MAE
result2 = np.mean(y_test == y_pred)

#%% make training set for layer 1
index_1 = []
for i in range(len(train_label)):
    if train_label[i,1] == train_label[i,2] and train_label[i,1] == train_label[i,3]:
        index_1.append(i)

index_2 = []
for i in range(len(train_label)):
    if train_label[i,0] == train_label[i,2] and train_label[i,0] == train_label[i,3]:
        index_2.append(i)

index_3 = []
for i in range(len(train_label)):
    if train_label[i,0] == train_label[i,1] and train_label[i,0] == train_label[i,3]:
        index_3.append(i)
        
index_4 = []
for i in range(len(train_label)):
    if train_label[i,0] == train_label[i,1] and train_label[i,0] == train_label[i,2]:
        index_4.append(i)
        
#%% make train data
train_1 = train_data[index_1,:]
train_2 = train_data[index_2,:]
train_3 = train_data[index_3,:]
train_4 = train_data[index_4,:]

label_1 = train_label[index_1,0]
label_2 = train_label[index_2,1]
label_3 = train_label[index_3,2]
label_4 = train_label[index_4,3]

#%% Grid CV for layer 1
parameters = {'max_depth': range(15,20,5), 'n_estimators': range(10,20,10)}
rf_2 = RandomForestRegressor()
clf_2 = GridSearchCV(rf_2, parameters,n_jobs=-1,cv=10,scoring='neg_mean_absolute_error')
clf_2.fit(train_1,label_1)


#%%

clf = RandomForestRegressor(max_depth=30, n_estimators=30, n_jobs=-1)
clf.fit(train_n, train_label)
predicted = clf.predict(test)

sample.layer_1 = predicted[:,0]
sample.layer_2 = predicted[:,1]
sample.layer_3 = predicted[:,2]
sample.layer_4 = predicted[:,3]

sample.to_csv('submit.csv',index=False)

#%% 
result = fft(train_n[-1,:])
plt.plot(train_n[-1,:])
plt.show()