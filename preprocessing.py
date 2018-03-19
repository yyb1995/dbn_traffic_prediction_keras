'''
Pre processing the network traffic data.
Input: 1*1008 data
Output:train data and test data
'''

import numpy as np


def generate_data(dataset,n_features,testnum):
    dataset_len = dataset.shape[1]
    testnum = int(testnum)
    x_train = np.zeros([dataset_len - n_features - testnum,n_features])
    x_test = np.zeros([testnum,n_features])
    #y_train = np.zeros([1008 - n_features - testnum,1])
    for i in range(dataset_len - n_features - testnum):
        x_train[i,:] = dataset[0,i:i + n_features]
    for i in range(testnum):
        x_test[i,:] = dataset[0,(i + dataset_len - testnum - n_features):(i + dataset_len - testnum)]
    #y_train = dataset[0,n_features:dataset_len - testnum - n_features]
    y_train = dataset[0, n_features:dataset_len - testnum]
    y_test = dataset[0,dataset_len - testnum:dataset_len]
    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    dataall = np.array([i for i in range(100)])[np.newaxis,:]
    a,b,c,d = generate_data(dataall,10,8)

