import numpy as np


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+ self.decay * self.iterations))
    
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weights_update = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weights_update
            biases_update =  self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = biases_update
        else:
            weights_update = -self.current_learning_rate * layer.dweights
            biases_update = - self.current_learning_rate * layer.dbiases

        layer.weights += weights_update
        layer.biases += biases_update

    def post_update_params(self):
        self.iterations += 1