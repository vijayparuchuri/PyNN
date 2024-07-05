import numpy as np

class Activation_Sigmoid:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        exponentiated_inputs = np.exp(-inputs)
        self.output = 1 / (1 + exponentiated_inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues  * (1 - self.output) * self.output
        
    def predictions(self, outputs):
        return (outputs > 0.5) * 1