import numpy as np

class SGD:
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


class AdaGrad:
    def __init__(self, learning_rate=0.01, decay=1e-4, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self. iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))
        
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        weight_update = - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        bias_update = - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += weight_update
        layer.biases += bias_update

    def post_update_params(self):
        self.iterations += 1


class RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., rho=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.rho = rho
        self.iterations = 0
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon


    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * 1./(1.+self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Adam:
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        self.current_learning_rate = self.learning_rate * 1./(1.+self.decay * self.iterations)
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta_1 * layer.weight_momentum + (1 - self.beta_1) * layer.dweights
        layer.bias_momentum = self.beta_1 * layer.bias_momentum + (1 - self.beta_1) * layer.dbiases

        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta_1**(self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1