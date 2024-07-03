import numpy as np


class Layer_Conv2D:
    def __init__(self, kernel_size):
        h, w = kernel_size
        self.V = 0.01 * np.random.randn(h, w)
        self.bias = np.zeros(1)
        
    def forward(self, input):
        h, w = self.V.shape
        self.output = np.empty((input.shape[0] - h + 1, input.shape[0] - w + 1))
        
        for column in range(self.output.shape[1]):
            for row in range(self.output.shape[0]):
                sub_array = input[row:row+h, column:column+w]
                self.output[row, column] = np.sum(sub_array*self.V) + self.bias