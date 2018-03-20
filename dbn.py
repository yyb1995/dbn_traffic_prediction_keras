import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from sklearn.neural_network import BernoulliRBM


class DBN():
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_layer,
            learning_rate_rbm=0.0001,
            batch_size_rbm=100,
            n_epochs_rbm=30,
            verbose_rbm=1,
            random_seed_rbm=1300,
            activation_function_nn='relu',
            learning_rate_nn=0.005,
            batch_size_nn=100,
            n_epochs_nn=10,
            verbose_nn=1,
            decay_rate=0):

            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.hidden_layer = hidden_layer
            self.learning_rate_rbm = learning_rate_rbm
            self.batch_size_rbm = batch_size_rbm
            self.n_epochs_rbm = n_epochs_rbm
            self.verbose_rbm = verbose_rbm
            self.random_seed = random_seed_rbm
            self.activation_function_nn = activation_function_nn
            self.learning_rate_nn = learning_rate_nn
            self.batch_size_nn = batch_size_nn
            self.n_epochs_nn = n_epochs_nn
            self.verbose_nn = verbose_nn
            self.decay_rate = decay_rate
            self.weight_rbm = []
            self.bias_rbm = []
            self.test_rms = 0
            self.result = []
            self.model = Sequential()

    def pretraining(self):
        input_layer = self.x_train
        for i in range(len(self.hidden_layer)):
            print("DBN Layer {0} Pre-training".format(i + 1))
            rbm = BernoulliRBM(n_components=self.hidden_layer[i],
                               learning_rate=self.learning_rate_rbm,
                               batch_size=self.batch_size_rbm,
                               n_iter=self.n_epochs_rbm,
                               verbose=self.verbose_rbm,
                               random_state=self.verbose_rbm)
            rbm.fit(input_layer)
            # size of weight matrix is [input_layer, hidden_layer]
            self.weight_rbm.append(rbm.components_.T)
            self.bias_rbm.append(rbm.intercept_hidden_)
            input_layer = rbm.transform(input_layer)
        print('Pre-training finish.')

    def finetuning(self):
        print('Fine-tuning start.')

        for i in range(0, len(self.hidden_layer)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn,
                                     input_dim=self.x_train.shape[1]))
            elif i >= 1:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function_nn))
            else:
                pass
            layer = self.model.layers[i]
            layer.set_weights([self.weight_rbm[i], self.bias_rbm[i]])
        if(self.y_train.ndim == 1):
            self.model.add(Dense(1, activation=None, kernel_regularizer=regularizers.l2(0.01)))
        else :
            self.model.add(Dense(self.y_train.shape[1], activation=None))

        sgd = SGD(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='mse',
                      optimizer=sgd,
                      )
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size_nn,
                       epochs=self.n_epochs_nn, verbose=self.verbose_nn)
        print('Fine-tuning finish.')
        self.test_rms = self.model.evaluate(self.x_test, self.y_test)
        self.result = np.array(self.model.predict(self.x_test))

    def predict(self, series):
        return np.array(self.model.predict(series))


