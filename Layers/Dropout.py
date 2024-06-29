import numpy as np

class Layer_Dropout:
    
    def __init__(self, dropout_rate):
        self.dropout_rate = 1 - dropout_rate
    
    def forward(self, inputs):
        self.inputs = inputs
        self.mask = np.random.binomial(1, self.dropout_rate, inputs.shape) / self.dropout_rate 
        self.output = inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask