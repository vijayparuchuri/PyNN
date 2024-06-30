from sklearn.datasets import load_iris
from keras.api.losses import BinaryCrossentropy
import pandas as pd
import numpy as np
from Layers.Dense import Layer_Dense
from Layers.Dropout import Layer_Dropout
from Activations.ReLU import Activation_ReLU
from Activations.Softmax import Activation_Softmax
from Activations.Sigmoid import Activation_Sigmoid
from Losses.CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from Losses.BinaryCrossEntropy import Loss_BinaryCrossEntropy
from Losses.ActivationLossDerivative import Activation_Softmax_Loss_CategoricalCrossEntropy
from Optimziers.Optimizers import SGD, AdaGrad, RMSProp, Adam
from nnfs.datasets import spiral_data

# iris = load_iris()

# df = pd.DataFrame(data = iris.data, columns=iris.feature_names)

# df['targets'] = iris.target

# X = df.drop('targets', axis=1).values
# y = df['targets'].values

X, y = spiral_data(100, 2)

y = y.reshape(-1, 1)

# X = np.array([[1, 2, 3, 2.5],
# [2., 5., -1., 2],
# [-1.5, 2.7, 3.3, -0.8]])
# y = np.array([[0,0,1], [1,0,0], [0,1,0]])

# Forward Pass

#Initializations
dense1  = Layer_Dense(2, 256, weight_regularizer_L2 = 5e-4, bias_regularizer_L2=5e-4)
dropout = Layer_Dropout(0.1)
dense2 = Layer_Dense(256, 1)
activation1 = Activation_ReLU()
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossEntropy()
loss_sk = BinaryCrossentropy(from_logits=False)
# activation_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
# optimizer = SGD(decay=1e-3, momentum=0.99, learning_rate=.1)
# optimizer = AdaGrad(0.1, decay=1e-4, epsilon=1e-4)
# optimizer = RMSProp(0.02, decay=1e-5, rho=0.9, epsilon=1e-7)
optimizer = Adam(learning_rate=0.01, decay=1e-4)

#Running the forward pass
for i in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout.forward(activation1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)
    
    
    # data_loss = activation_loss.forward(dense2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    
    loss = data_loss + regularization_loss
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    if i % 100 == 0:
        print(f'Epoch: {i}....Loss: {loss:.3f} (data_loss: {data_loss:.3f}....reg_loss: {regularization_loss:.3f})....Accuracy: {accuracy}')
    # predictions = np.argmax(activation_loss.output, axis = 1)
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    # accuracy = np.mean(predictions == y)
    # if i % 100 == 0:
    #     print(f'Epoch: {i}....Accuracy: {accuracy:.3f}....Loss: {loss:.3f} (data_loss:{data_loss:.3f}...reg_loss:{regularization_loss:.3f})....lr: {optimizer.current_learning_rate:.3f}')

#Backward Pass
    # activation_loss.backward(activation_loss.output, y)
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    dropout.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    ############Validation dataset##############
    X_test, y_test = spiral_data(100, 2)
    y_test = y_test.reshape(-1, 1)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dloss =  loss_function.calculate(activation2.output, y)
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_test)
    # predictions = np.argmax(activation_loss.output, axis=1)
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    # accuracy = np.mean(predictions==y)
    if i % 100 == 0:
        print(f'Validation Accuracy:{accuracy:.4f}, Validation Loss:{dloss:.4f}')
