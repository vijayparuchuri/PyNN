import numpy as np

# Softmax activation class
class Activation_Softmax:
    def forward(self, input):
        input = input-np.max(input, axis=1, keepdims=1)
        exponetiated = np.exp(input)
        self.output = exponetiated/np.sum(exponetiated, axis=1, keepdims=True)
