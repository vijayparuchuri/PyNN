from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from Dense import Layer_Dense
from ReLU import Activation_ReLU
from Softmax import Activation_Softmax
from CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from ActivationLossDerivative import Activation_Softmax_Loss_CategoricalCrossEntropy
from Optimizers import SGD, AdaGrad, RMSProp, Adam

# iris = load_iris()

# df = pd.DataFrame(data = iris.data, columns=iris.feature_names)

# df['targets'] = iris.target

# X = df.drop('targets', axis=1).values
# y = df['targets'].values

def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


X, y = spiral_data(100, 3)

# X = np.array([[1, 2, 3, 2.5],
# [2., 5., -1., 2],
# [-1.5, 2.7, 3.3, -0.8]])
# y = np.array([[0,0,1], [1,0,0], [0,1,0]])

# Forward Pass

#Initializations
dense1  = Layer_Dense(2, 64, weight_regularizer_L2 = 5e-4, bias_regularizer_L2=5e-4)
dense2 = Layer_Dense(64, 3)
activation1 = Activation_ReLU()
activation_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
# optimizer = SGD(decay=1e-3, momentum=0.99, learning_rate=.1)
# optimizer = AdaGrad(0.1, decay=1e-4, epsilon=1e-4)
# optimizer = RMSProp(0.02, decay=1e-5, rho=0.9, epsilon=1e-7)
optimizer = Adam(learning_rate=0.05, decay=5e-7)

#Running the forward pass
for i in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss = activation_loss.forward(dense2.output, y)
    regularization_loss = activation_loss.loss.regularization_loss(dense1) + activation_loss.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    predictions = np.argmax(activation_loss.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if i % 100 == 0:
        print(f'Epoch: {i}....Accuracy: {accuracy:.3f}....Loss: {loss:.3f} (data_loss:{data_loss:.3f}...reg_loss:{regularization_loss:.3f})....lr: {optimizer.current_learning_rate:.3f}')

#Backward Pass
    activation_loss.backward(activation_loss.output, y)
    dense2.backward(activation_loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    ############Validation dataset##############
    X, y = spiral_data(100, 3)
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss =  activation_loss.forward(dense2.output, y)
    predictions = np.argmax(activation_loss.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if i % 100 == 0:
        print(f'Validation Accuracy:{accuracy:.4f}, Validation Loss:{loss:.4f}')
