import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as k_backend
from random_forest import WaveletsForestRegressor
from keras_mnist import read_data

# Load the data
x_train, x_test, y_train, y_test, input_shape = read_data()

model = load_model('myModel.h5')


def calc_smoothness(x, y):
    wfr = WaveletsForestRegressor(regressor='random_forest', criterion='mse', depth=9, trees=5)
    wfr.fit(x, y)
    alpha, n_wavelets, errors = wfr.evaluate_smoothness(m=100)
    return alpha


def plot_vec(x=0, y=None, title='', xaxis='', yaxis=''):
    if x == 0:
        x = range(1, len(y) + 1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def main():
    alpha_vec = np.zeros((len(model.layers), ))
    for idx, layer in enumerate(model.layers):
        print('Calculating smoothness parameters for layer '+str(idx)+'.')
        get_layer_output = k_backend.function([model.layers[0].input],
                                              [model.layers[idx].output])
        layer_output = get_layer_output([x_train])[0]
        alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).transpose(),
                                         y_train)
    plot_vec(y=alpha_vec, title='Smoothness over layers', xaxis='Alpha', yaxis='#Layer')


if '__main__' == __name__:
    main()
