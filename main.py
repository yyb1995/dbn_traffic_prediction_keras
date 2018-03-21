from sklearn.preprocessing import MinMaxScaler
from preprocessing import generate_data, generate_data_nstep
from sklearn.metrics.regression import mean_squared_error
from mle import generateData
import numpy as np
import dbn
import dwt
import matplotlib.pyplot as plt


def data_import(filepath, scale):
    dataset_5 = np.load(filepath)
    row, col = dataset_5.shape
    if scale == 1:
        return dataset_5
    else:
        return np.reshape(np.average(np.reshape(dataset_5, (int(row * col / scale), scale)),
                                     axis=1), (row, int(col / scale)))


def predict_with_dwt(dataset, testnum, featurenum):
    ca, cd = dwt.dwt(dataset)
    ca_matrix = ca[np.newaxis, :]
    print('DWT finish.')
    x_train, x_test, y_train, y_test = generate_data(ca_matrix, int(testnum / 2), featurenum)
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    dbn1 = dbn.DBN(x_train=x_train,
                   y_train=y_train,
                   x_test=x_test,
                   y_test=y_test,
                   hidden_layer=[250],
                   learning_rate_rbm=0.0005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=150,
                   n_epochs_nn=1500,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    ca_pred = dbn1.result[:, 0]
    print('Lowpass coefficient estimation finish.')
    mu, sigma_2, cd_pred = generateData(cd[0:len(cd) - int(testnum / 2)],
                                        outputnum=int(testnum / 2))
    print('Highpass coefficient estimation finish.')
    dataset_pred = dwt.idwt(ca_pred, cd_pred)
    print('IDWT finish.')
    dataset_test = dataset[len(dataset) - testnum:len(dataset)]
    ca_test, cd_test = dwt.dwt(dataset_test)
    plt.figure(figsize=(12, 9), dpi=100)
    plt.subplot(3, 1, 1)
    plt.plot(ca_test)
    plt.plot(ca_pred)
    plt.legend(['lowpass_real', 'lowpass_prediction'], loc='upper right')
    plt.title('lowpass coefficient prediction result', fontsize=16)
    plt.subplot(3, 1, 2)
    plt.plot(cd_test)
    plt.plot(cd_pred)
    plt.legend(['highpass_real', 'highpass_prediction'], loc='upper right')
    plt.title('highpass coefficient prediction result', fontsize=16)
    plt.subplot(3, 1, 3)
    mse = mean_squared_error(dataset_pred, dataset_test)
    plt.plot(dataset_test)
    plt.plot(dataset_pred)
    plt.legend(['dataset_real', 'dataset_prediction'], loc='upper right')
    plt.title('sequence prediction result', fontsize=16)
    plt.xlabel('MSE = %f' % mse)
    plt.draw()
    #plt.show()
    return dataset_pred, mse


def predict_without_dwt(dataset, testnum, featurenum):
    dataset = dataset[np.newaxis, :]
    x_train, x_test, y_train, y_test = generate_data(dataset, testnum, featurenum)
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    dbn1 = dbn.DBN(x_train=x_train,
                   y_train=y_train,
                   x_test=x_test,
                   y_test=y_test,
                   hidden_layer=[250],
                   learning_rate_rbm=0.0005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=150,
                   n_epochs_nn=1500,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dataset_pred = dbn1.result[:, 0]
    dataset_test = dataset[0, dataset.shape[1] - testnum:dataset.shape[1]]
    mse = mean_squared_error(dataset_pred, dataset_test)
    plt.figure(figsize=(12, 9), dpi=100)
    plt.plot(dataset_test)
    plt.plot(dataset_pred)
    plt.legend(['dataset_real', 'dataset_prediction'], loc='upper right')
    plt.title('sequence prediction result', fontsize=16)
    plt.xlabel('MSE = %f' % mse)
    plt.draw()
    #plt.show()
    return dataset_pred, mse


def direct(dataset, testnum, featurenum):
    dataset = dataset[np.newaxis, :]
    x_train_one, x_test, y_train_one, y_test = generate_data(dataset, testnum, featurenum)
    x_train_two = x_train_one[0:x_train_one.shape[0] - 1, :]
    y_train_two = y_train_one[1:y_train_one.shape[0]]
    x_test_two = x_test[::2, :]
    y_test_two = y_test[1::2]
    x_test_one = x_test[::2, :]
    y_test_one = y_test[::2]
    min_max_scaler1 = MinMaxScaler()
    x_train_one = min_max_scaler1.fit_transform(x_train_one)
    x_test_one = min_max_scaler1.transform(x_test_one)
    min_max_scaler2 = MinMaxScaler()
    x_train_two = min_max_scaler2.fit_transform(x_train_two)
    x_test_two = min_max_scaler2.transform(x_test_two)
    dbn1 = dbn.DBN(x_train=x_train_one,
                   y_train=y_train_one,
                   x_test=x_test_one,
                   y_test=y_test_one,
                   hidden_layer=[250],
                   learning_rate_rbm=0.0005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=200,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dataset_pred_one = dbn1.result[:, 0]
    dbn2 = dbn.DBN(x_train=x_train_two,
                   y_train=y_train_two,
                   x_test=x_test_two,
                   y_test=y_test_two,
                   hidden_layer=[250],
                   learning_rate_rbm=0.0005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=200,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn2.pretraining()
    dbn2.finetuning()
    dataset_pred_two = dbn2.result[:, 0]
    dataset_pred = []
    for i in range(len(dbn2.result[:, 0])):
        dataset_pred.append(dataset_pred_one[i])
        dataset_pred.append(dataset_pred_two[i])
    return dataset_pred


def recursive(dataset, testnum, featurenum):
    dataset = dataset[np.newaxis, :]
    x_train, x_test, y_train, y_test = generate_data(dataset, testnum, featurenum)
    x_test_one = x_test[::2, :]
    y_test_one = y_test[::2]
    y_test_two = y_test[1::2]
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test_one = min_max_scaler.transform(x_test_one)
    dataset_pred = []
    dbn1 = dbn.DBN(x_train=x_train,
                   y_train=y_train,
                   x_test=x_test_one,
                   y_test=y_test_one,
                   hidden_layer=[250],
                   learning_rate_rbm=0.0005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=200,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dataset_pred_one = dbn1.result[:, 0]
    x_test_two = np.delete(x_test[::2, :], 0, axis=1)
    x_test_two = np.hstack((x_test_two, dbn1.result.reshape(len(dbn1.result), 1)))
    x_test_two = min_max_scaler.transform(x_test_two)
    dataset_pred_two = dbn1.predict(x_test_two)
    for i in range(len(dataset_pred_one)):
        dataset_pred.append(dataset_pred_one[i])
        dataset_pred.append(dataset_pred_two[i])
    return dataset_pred


def multioutput(dataset, testnum, featurenum, nstep):
    dataset = dataset[np.newaxis, :]
    x_train, x_test, y_train, y_test = generate_data_nstep(dataset, testnum, featurenum, nstep)
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    dbn1 = dbn.DBN(x_train=x_train,
                   y_train=y_train,
                   x_test=x_test,
                   y_test=y_test,
                   hidden_layer=[250],
                   learning_rate_rbm=0.005,
                   batch_size_rbm=150,
                   n_epochs_rbm=200,
                   verbose_rbm=1,
                   random_seed_rbm=500,
                   activation_function_nn='tanh',
                   learning_rate_nn=0.005,
                   batch_size_nn=300,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dataset_pred = dbn1.result.reshape(1, testnum)
    return dataset_pred[0, :]


# main function section

# import the test data
dataset = data_import('E:\\北航\\研究生\\Abilene TM\\Abilene_npy\\20040501_144_2.npy', scale=1)
#dataset2 = data_import('E:\\北航\\研究生\\Abilene TM\\Abilene_npy\\20040508_144_2.npy', scale=2)
#dataset3 = data_import('E:\\北航\\研究生\\Abilene TM\\Abilene_npy\\20040515_144_2.npy', scale=1)
#dataset4 = data_import('E:\\北航\\研究生\\Abilene TM\\Abilene_npy\\20040522_144_2.npy', scale=4)
dataset_dbn = dataset[0, 0:1875] / 1000

#dataset_dbn = np.hstack((dataset, dataset2))[0, 0:1800] / 500
#plt.plot(dataset_dbn)
#plt.show()



# set some model parameters
testnum = 50
featurenum = 50
dataset_test = dataset_dbn[len(dataset_dbn) - testnum:len(dataset_dbn)]


'''
Section 1
Simple implement of DBN-NN prediction model
testnum = 50
'''
'''
dataset_pred_dwt, mse_dwt = predict_with_dwt(dataset_dbn, testnum, featurenum)
plt.show()
'''




'''
Section 2
test different resolution ratio model performance
'''






'''
Section 3
compare mse on whether using dwt method
testnum = 50
'''
'''

dataset_pred_dwt, mse_dwt = predict_with_dwt(dataset_dbn, testnum, featurenum)
dataset_pred_nodwt, mse_nodwt = predict_without_dwt(dataset_dbn, testnum, featurenum)
plt.figure(figsize=(12, 9), dpi=100)
plt.plot(dataset_test)
plt.plot(dataset_pred_dwt)
plt.plot(dataset_pred_nodwt)
plt.legend(['dataset_real', 'dataset_dwt', 'dataset_nodwt'])
plt.xlabel('mse_dwt : %f\nmse_nodwt : %f' % (mse_dwt, mse_nodwt))
plt.title('Comparison on whether to use dwt transform')
plt.show()

'''

'''
Section 4
compare the three multi-step prediction method. For simplicity, omit the dwt process.
step = 2
testnum = 50
'''

'''

pred_direct = direct(dataset_dbn, testnum=testnum, featurenum=featurenum)
mse_direct = mean_squared_error(pred_direct, dataset_test)
print('Direct prediction finish.')
pred_recursive = recursive(dataset_dbn, testnum=testnum, featurenum=featurenum)
mse_recursive = mean_squared_error(pred_recursive, dataset_test)
print('Recursive prediction finish.')
pred_multioutput = multioutput(dataset_dbn, testnum=testnum, featurenum=featurenum, nstep=2)
mse_multioutput = mean_squared_error(pred_multioutput, dataset_test)
print('Multioutput prediction finish.')
plt.figure(figsize=(12, 9), dpi=100)
plt.plot(dataset_test)
plt.plot(pred_direct)
plt.plot(pred_recursive)
plt.plot(pred_multioutput)
plt.legend(['dataset_real', 'direct', 'recursive', 'multioutput'])
plt.xlabel('mse_independent : %f\nmse_recursive : %f\nmse_multioutput : %f' % (mse_direct, mse_recursive, mse_multioutput))
plt.title('Comparison on three multi-step prediction method')
plt.show()

'''



