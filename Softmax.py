import numpy as np

# Softmax activation class
class Activation_Softmax:
    def forward(self, input):
        self.input = input
        input = input-np.max(input, axis=1, keepdims=True)
        exponetiated = np.exp(input)
        self.output = exponetiated/np.sum(exponetiated, axis=1, keepdims=True)
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(single_dvalue, jacobian_matrix)