'''
Pre processing the network traffic data.
Input: 1*1008 data
Output:train data and test data
'''

import numpy as np


def generate_data(dataset, testnum, n_features):
    dataset_len = dataset.shape[1]
    testnum = int(testnum)
    x_train = np.zeros([dataset_len - n_features - testnum, n_features])
    x_test = np.zeros([testnum,n_features])
    for i in range(dataset_len - n_features - testnum):
        x_train[i,:] = dataset[0,i:i + n_features]
    for i in range(testnum):
        x_test[i,:] = dataset[0,(i + dataset_len - testnum - n_features):(i + dataset_len - testnum)]
    #y_train = dataset[0,n_features:dataset_len - testnum - n_features]
    y_train = dataset[0, n_features:dataset_len - testnum]
    y_test = dataset[0,dataset_len - testnum:dataset_len]
    return x_train,x_test,y_train,y_test


def generate_data_nstep(dataset,  testnum, n_features, nstep):
    dataset_len = dataset.shape[1]
    testnum_nstep = int(testnum / nstep)
    x_train = np.zeros([dataset_len - testnum - n_features, n_features])
    y_train = np.zeros([dataset_len - testnum - n_features, nstep])
    x_test = np.zeros([testnum_nstep, n_features])
    y_test = np.zeros([testnum_nstep, nstep])
    for i in range(dataset_len - n_features - testnum):
        x_train[i,:] = dataset[0,i:i + n_features]
    for i in range(testnum_nstep):
        x_test[i,:] = dataset[0,(i * nstep + dataset_len - testnum - n_features):(i * nstep + dataset_len - testnum)]
    for i in range(dataset_len - n_features - testnum):
        y_train[i,:] = dataset[0, i + n_features:i + n_features + nstep]
    for i in range(testnum_nstep):
        y_test[i,:] = dataset[0, i * nstep + dataset_len - testnum:(i + 1) * nstep + dataset_len - testnum]
    return x_train, x_test, y_train, y_test
if __name__ == '__main__':
    dataall = np.array([i for i in range(100)])[np.newaxis,:]
    a,b,c,d = generate_data(dataall,10,8)

