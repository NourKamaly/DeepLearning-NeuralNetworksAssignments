import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, class1, class2, lr=0.01, epochs=1000, add_bias=True):
        self.lr = lr
        self.epochs = epochs
        self.activation_function = self.signum_f
        self.weights = None
        self.bias = 0
        self.add_bias = add_bias
        self.class1 = class1
        self.class2 = class2

    def fit(self, X, Y, Species):
        M_samples, N_features = X.shape

        # init weights
        w = np.zeros(N_features)
        random.seed(0)
        for i in range(N_features):
            w[i] = random.random()
        self.weights = w
        if (self.add_bias):
            self.bias = random.random()

        # converting Y categorical values  to 1 and -1
        y_converted = Y.to_numpy()
        y = np.array([1 if i == Species[self.class1] else -1 for i in y_converted])

        # Training loop
        x = X.to_numpy()
        for _ in range(self.epochs):
            for indx, x_i in enumerate(x):
                linear_output = np.dot(x_i, self.weights)
                if (self.add_bias):
                    linear_output += self.bias
                y_predict = self.activation_function(linear_output)

                update = self.lr * (y[indx] - y_predict)
                self.weights += update * x_i
                if (self.add_bias):
                    self.bias += update

    # X is np array
    def predict(self, X):
        linear_out = np.dot(X, self.weights)
        if (self.add_bias):
            linear_out += self.bias
        y_predicted = self.activation_function(linear_out)
        return y_predicted

    def signum_f(self, x):
        return np.where(x >= 0, 1, -1)