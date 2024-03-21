import numpy as np

#The base ReLU activation class
class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


