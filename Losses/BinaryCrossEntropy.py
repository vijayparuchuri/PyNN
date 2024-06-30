import numpy as np
from Losses.CategoricalCrossEntropy import Loss

class Loss_BinaryCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        clipped_y_preds = np.clip(y_pred, 1e-7, 1-1e-7)
        
        output = - (y_true * np.log(clipped_y_preds) + (1 - y_true) * np.log(1 - clipped_y_preds))
        
        return output
        
    def backward(self, y_pred, y_true):
        samples = len(y_true)
        outputs = len(y_true[0])
        clipped_y_preds =  np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = - (y_true / clipped_y_preds - (1 - y_true) / (1 - clipped_y_preds)) / outputs
        self.dinputs = self.dinputs / samples
        
        