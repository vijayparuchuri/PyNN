from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from keras.api.losses import BinaryCrossentropy
import pandas as pd
import numpy as np
from Layers.Dense import Layer_Dense
from Layers.Dropout import Layer_Dropout
from Activations.ReLU import Activation_ReLU
from Activations.Softmax import Activation_Softmax
from Activations.Sigmoid import Activation_Sigmoid
from Activations.LinearActivation import Activation_Linear
from Losses.CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from Losses.BinaryCrossEntropy import Loss_BinaryCrossEntropy
from Losses.MeanSquaredError import Loss_MeanSquaredError
from Losses.MeanAbsoluteError import Loss_MeanAbsoluteError
from Losses.ActivationLossDerivative import Activation_Softmax_Loss_CategoricalCrossEntropy
from Optimziers.Optimizers import SGD, AdaGrad, RMSProp, Adam
from nnfs.datasets import spiral_data, sine_data
from sklearn.metrics import mean_squared_error

X, y = sine_data()


dense1 = Layer_Dense(1, 64)
dropout = Layer_Dropout(0.1)
dense2 = Layer_Dense(64, 64)
dense3 = Layer_Dense(64, 1)
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Linear()
loss_fn = Loss_MeanSquaredError()
optimizer = Adam(learning_rate=0.001, decay=1e-3)

epochs = 10001
precision = np.std(y) / 250

for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout.forward(activation1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_fn.calculate(activation3.output, y)
    
    regularization_loss = loss_fn.regularization_loss(dense1) + loss_fn.regularization_loss(dense2)
    
    total_loss = data_loss + regularization_loss
    
    predictions = activation3.output
    accuracy = np.mean(np.absolute(y - predictions) < precision) 
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}....Accuracy: {accuracy}....Loss: {total_loss:.3f} (Data_Loss: {data_loss:.3f}, Regularization_loss: {regularization_loss:.3f})')
    
    
    loss_fn.backward(activation3.output, y)
    activation3.backward(loss_fn.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    dropout.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
    

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X, y)
plt.plot(X_test, activation3.output)

plt.show()

    