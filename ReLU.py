import numpy as np

#The base ReLU activation class
class Activation_ReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    def backward(self, dvalue):
        self.dinputs = self.dvlaues.copy()
        self.dinputs[self.dinputs < 0] = 0
