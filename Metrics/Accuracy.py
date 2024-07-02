import numpy as np



class Accuracy:
    def calculate(self, predictions, y):
        comparisions = self.compare(predictions, y)
        
        accuracy = np.mean(comparisions)
        
        return accuracy

class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        self.precision = None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    def compare(self, predictions, y):
        return np.absolute(y - predictions) < self.precision
    
class Accuracy_Categorical(Accuracy):
    def __init__(self, binary = False):
        self.binary = binary
    
    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    
