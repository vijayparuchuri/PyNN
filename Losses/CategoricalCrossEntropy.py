import numpy as np

class Loss:
    def calculate(self, output, y):
        loss = np.mean(self.forward(output, y))
        return loss
    def regularization_loss(self, layer):
        regularization_loss = 0
        
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
        
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
        
        if layer.bias_regularaizer_L1 > 0:
            regularization_loss += layer.bias_regularaizer_L1 * np.sum(np.abs(layer.biases))
        
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.bias_regularizer_L2 * layer.bias_regularizer_L2)
        
        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelyhood = -np.log(correct_confidences)
        return negative_log_likelyhood
        
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = - y_true / dvalues
        self.dinputs = self.dinputs / samples
        