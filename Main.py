from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from Dense import Layer_Dense
from ReLU import Activation_ReLU
from Softmax import Activation_Softmax
from CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns=iris.feature_names)

df['targets'] = iris.target

X = df.drop('targets', axis=1).values
y = df['targets'].values
# X = np.array([[1, 2, 3, 2.5],
# [2., 5., -1., 2],
# [-1.5, 2.7, 3.3, -0.8]])
# y = np.array([[0,0,1], [1,0,0], [0,1,0]])

# Forward Pass

#Initializations
dense1  = Layer_Dense(4, 64)
dense2 = Layer_Dense(64, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss = Loss_CategoricalCrossEntropy()

#Running the forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(loss.calculate(activation2.output, y))


#Backward Pass
loss.backward(activation2.output, y)
activation2.backward(loss.dinputs)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)