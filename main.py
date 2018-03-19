from sklearn.preprocessing import MinMaxScaler
from preprocessing import generate_data
from sklearn.metrics.regression import mean_squared_error
from mle import generateData
import numpy as np
import dbn
import dwt
import matplotlib.pyplot as plt



def data_import(filepath, scale):
    dataset_5 = np.load(filepath)
    row,col = dataset_5.shape
    if scale == 1:
        return dataset_5
    else:
        return np.reshape(np.average(np.reshape(dataset_5,(int(row * col / scale), scale)),
                axis=1),(row, int(col / scale)))




def predict_with_dwt(dataset_dbn, testnum, featurenum):
    ca, cd = dwt.dwt(dataset_dbn)
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
                   batch_size_nn=200,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    #print('RMS on Testing set:%f' % dbn1.test_rms)
    ca_pred = dbn1.result
    print('Lowpass coefficient estimation finish.')
    mu, sigma_2, cd_pred = generateData(cd[0:len(cd) - int(testnum / 2)],
                                            outputnum=int(testnum / 2))
    print('Highpass coefficient estimation finish.')
    dataset_pred = dwt.idwt(ca_pred, cd_pred)
    print('IDWT finish.')
    dataset_test = dataset_dbn[len(dataset_dbn) - testnum:len(dataset_dbn)]
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
    plt.show()
    return dataset_pred, mse


def predict_without_dwt(dataset_dbn, testnum, featurenum):
    dataset_dbn = dataset_dbn[np.newaxis, :]
    x_train, x_test, y_train, y_test = generate_data(dataset_dbn, testnum, featurenum)
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
                   batch_size_nn=200,
                   n_epochs_nn=300,
                   verbose_nn=1,
                   decay_rate=0)
    dbn1.pretraining()
    dbn1.finetuning()
    dataset_pred = dbn1.result
    dataset_test = dataset_dbn[len(dataset_dbn) - testnum:len(dataset_dbn)]
    mse = mean_squared_error(dataset_pred, dataset_test)
    plt.figure(figsize=(12, 9), dpi=100)
    plt.plot(dataset_test)
    plt.plot(dataset_pred)
    plt.legend(['dataset_real', 'dataset_prediction'], loc='upper right')
    plt.title('sequence prediction result', fontsize=16)
    plt.xlabel('MSE = %f' % mse)
    plt.show()
    return dataset_pred, mse
# main function section

dataset = data_import('E:\\北航\\研究生\\Abilene TM\\Abilene_npy\\20040501_144_2.npy', scale=1)
dataset_dbn = dataset[0, :] / 1000

testnum = 50
featurenum = 50


#Compare mse on whether using dwt method



