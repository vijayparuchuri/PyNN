import numpy as np

# The base Dense layer with a forward and a backward method
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases

    def  backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(inputs.T, dvalues)
        self.dweights = np.sum(dvalues, axis=0, keepdims=True)

