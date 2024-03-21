import numpy as np

# Softmax activation class
class Activation_Softmax:
    def forward(self, input):
        samples = len(input)
        exponetiated = np.exp(input)
        self.output = exponetiated/np.sum(exponetiated, axis=1, keepdims=True)
