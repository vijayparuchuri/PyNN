import numpy as np
from CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from Softmax import Activation_Softmax

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.loss = Loss_CategoricalCrossEntropy()
        self.activation = Activation_Softmax()

    def forward(self, output, y):
        self.activation.forward(output)
        self.output = self.activation.output
        return self.loss.calculate(self.activation.output, y)

    def backward(self, dvalues, y):
        samples = len(dvalues)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y] -=1 
        self.dinputs = self.dinputs/samples
