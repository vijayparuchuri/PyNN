import numpy as np
from Losses.CategoricalCrossEntropy import Loss

class Loss_MeanSquaredError(Loss):
    
    def forward(self, y_pred, y_true):
        sample_mse =  np.mean(np.square(y_true - y_pred), axis=-1)

        return sample_mse
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        output = len(y_pred[0])
        self.dinputs = -( 2 * (y_true - y_pred) ) / output
        self.dinputs = self.dinputs/samples