import numpy as np


class MLP:
    def __init__(self, num_layers, num_neurons, add_bias, activation_fun, eta=0.01, epochs=1000):
        self.lr = eta
        self.epochs = epochs
        self.activation_function = activation_fun
        self.weights = None
        self.add_bias = add_bias
        self.layers = num_layers
        self.neurons = num_neurons

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def der_sigmoid(self, sigmoid):
        return sigmoid * (1 - sigmoid)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def der_tanh(self, tanh):
        return 1 - (tanh * tanh)

    def initialize_weights(self, L, architecture_layers, f_bias):
        """
        Arguments:
        L == num of layers
        architecture_layers == list containing the dimensions of each layer in our deep neural network
        f_bias == flag for bias

        Returns:
        weights == python dictionary containing ("W1", "b1", ..., "WL", "bL")
        """

        np.random.seed(0)
        weights = {}

        for l in range(1, L):

            weights['W' + str(l)] = np.random.randn(architecture_layers[l], architecture_layers[l - 1])

            if f_bias:
                weights['b' + str(l)] = np.zeros((architecture_layers[l], 1))

        return weights

    def forward_propagation(self, L, X, parameters, activation, f_bias):
        """
        Arguments:
        L == num of layers
        X == array of shape (n,1)
        parameters == dict of initialize_weights
        activation == activation function (sigmoid) or (tanh)
        f_bias == flag for bias


        Returns:
        Activaions == dict containing the result of activation functions of each layer in our deep neural network
        """

        Activaions = {}
        A = X
        Activaions['F' + str(0)] = A

        for l in range(1, L):
            A_prev = A

            Z = np.dot(parameters['W' + str(l)], A_prev)

            if f_bias:
                Z += parameters['b' + str(l)]

            if activation == "sigmoid":
                A = self.sigmoid(Z)
            elif activation == "tanh":
                A = self.tanh(Z)

            Activaions['F' + str(l)] = A

        return Activaions

    def backward_propagation(self, L, X, Y, parameters, activation, Activaions):
        """
        Arguments:
        L == num of layers
        X == array of shape (n,1)
        Y == Actual value (3,1)
        activation == activation function (sigmoid) or (tanh)
        Activaions == dict of activation values for each layer
        parameters == dict of initialize_weights

        Returns:
        gradients == dict containing the result of gradients for each layer in our deep neural network
        """

        gradients = {}
        Error = Y - Activaions['F' + str(L - 1)]

        if activation == "sigmoid":
            gradients['G' + str(L - 1)] = Error * self.der_sigmoid(Activaions['F' + str(L - 1)])  # (3,1)
        elif activation == "tanh":
            gradients['G' + str(L - 1)] = Error * self.der_tanh(Activaions['F' + str(L - 1)])

        for l in reversed(range(1, L - 1)):

            if activation == "sigmoid":
                gradients['G' + str(l)] = np.dot(parameters['W' + str(l + 1)].T,
                                                 gradients['G' + str(l + 1)]) * self.der_sigmoid(
                    Activaions['F' + str(l)])
            elif activation == "tanh":
                gradients['G' + str(l)] = np.dot(parameters['W' + str(l + 1)].T,
                                                 gradients['G' + str(l + 1)]) * self.der_tanh(Activaions['F' + str(l)])

        return gradients

    def update_parameters(self, L, Activaions, parameters, gradients, eta, f_bias):
        """
        Arguments:
        L == num of layers
        eta == learning rate
        f_bias == flag for bias
        Activaions == dict of activation values for each layer
        parameters == dict of initialize_weights
        gradients == dict containing the result of gradients for each layer in our deep neural network


        Returns:
        parameters -- python dictionary containing your updated parameters
        """

        for l in range(1, L):
            parameters["W" + str(l)] = parameters["W" + str(l)] + eta * np.dot(gradients["G" + str(l)],
                                                                               Activaions['F' + str(l - 1)].T)

            if f_bias:
                parameters["b" + str(l)] = parameters["b" + str(l)] + eta * gradients["G" + str(l)]

        return parameters

    def forward(self, L, X, parameters, activation, f_bias, prediction):

        A = X

        for l in range(1, L):
            A_prev = A

            Z = np.dot(parameters['W' + str(l)], A_prev)  # Z(l,1)

            if f_bias:
                Z += parameters['b' + str(l)]

            if activation == "sigmoid":
                A = self.sigmoid(Z)
            elif activation == "tanh":
                A = self.tanh(Z)

        Max_value = A.max()
        y_predicted = [0 if i < Max_value else 1 for i in A]
        prediction.append(y_predicted)

    def fit(self, X, Y):

        self.weights = self.initialize_weights(self.layers, self.neurons, self.add_bias)

        y = Y.to_numpy()
        x = X.to_numpy()

        for _ in range(self.epochs):
            for indx, x_i in enumerate(x):
                x_i = x_i.reshape((len(x_i), 1))
                y_i = y[indx]
                y_i = y_i.reshape((len(y_i), 1))

                Activaions = self.forward_propagation(self.layers, x_i, self.weights, self.activation_function,
                                                      self.add_bias)
                gradients = self.backward_propagation(self.layers, x_i, y_i, self.weights, self.activation_function,
                                                      Activaions)
                self.weights = self.update_parameters(self.layers, Activaions, self.weights, gradients, self.lr,
                                                      self.add_bias)

    def predict(self, X):
        x = X.to_numpy()
        prediction = []

        for x_i in x:
            x_i = x_i.reshape((len(x_i), 1))
            self.forward(self.layers, x_i, self.weights, self.activation_function, self.add_bias, prediction)

        return prediction