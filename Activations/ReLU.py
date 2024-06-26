import numpy as np

#The base ReLU activation class
class Activation_ReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
