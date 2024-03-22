import numpy as np

class Loss:
    def calculate(self, output, y):
        loss = np.mean(self.forward(output, y))
        return loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape):
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelyhood = -np.log(correct_confidences)
        return negative_log_likelyhood
        