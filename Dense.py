import numpy as np

# X = np.random.rand(3,4)
# y = np.array([[0,0,1], [1,0,0], [0,1,0]])


# The base Dense layer with a forward and a backward method
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases

# dense1 = Layer_Dense(4, 4)
#
# dense2 = Layer_Dense(4, 3)
#
# dense1.forward(X)
# dense2.forward(dense1.output)
#
# print(dense2.output)
