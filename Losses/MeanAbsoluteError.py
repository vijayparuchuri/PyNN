import numpy as np
from Losses.CategoricalCrossEntropy import Loss

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_mae = np.mean(np.abs(y_true - y_pred), axis=-1)
        
        return sample_mae
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        output = len(y_pred[0])
        self.dinputs = np.sign(y_true - y_pred) / output
        self.dinputs = self.dinputs/samples