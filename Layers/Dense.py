import numpy as np

# The base Dense layer with a forward and a backward method
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1 = 0, weight_regularizer_L2 = 0,
                bias_regularizer_L1 = 0, bias_regularizer_L2 = 0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])
        # Set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularaizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def  backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights<0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        
        if self.bias_regularaizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularaizer_L1 * dL1
        
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
        
        self.dinputs = np.dot(dvalues, self.weights.T)


