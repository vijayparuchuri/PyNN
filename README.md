# PyNN: A Mini Deep Learning Library

PyNN is a lightweight, open-source mini deep learning library built entirely with NumPy. It provides a flexible framework for building, training, and deploying neural networks, focusing on simplicity and hackability. The project is designed to help users understand the underlying mechanics of deep learning by implementing all components from mathematical first principles.

## Features

- **Custom Layers**:
  - Convolutional (Conv)
  - MaxPooling and AveragePooling
  - Dense and Linear layers
- **Loss Functions**:
  - CrossEntropy
  - BinaryCrossEntropy
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- **Activation Functions**:
  - ReLU
  - Sigmoid
  - Softmax
  - LinearActivation
- **Optimization Algorithms**:
  - Stochastic Gradient Descent (SGD)
  - RMSProp
  - Adam
  - AdaGrad
- **Keras-like Model API**:
  - Define models, add layers, compile them with optimizers and loss functions, and train them with minimal code.
- **More to come**
## Why PyNN?

PyNN is built for those who want to understand the inner workings of deep learning without relying on complex frameworks like TensorFlow or PyTorch. It is ideal for educational purposes as well as experimentation with custom neural network architectures.

## Installation

Clone the repository:
```
git clone https://github.com/vijayparuchuri/PyNN.git
cd PyNN
```
## Usage

Here’s an example of how to use PyNN to build and train a simple neural network:

```
from pynn import Model
from pynn.Layers import Layer_Dense,
From PyNN.Activations import ReLU, Softmax
from pynn.Losses import CrossEntropyLoss
from pynn.Optimizers import SGD
# Initialize model
model = Model()
# Add layers
model.add(Dense(input_dim=4, output_dim=8))
model.add(ReLU())
model.add(Dense(input_dim=8, output_dim=3))
model.add(Softmax())
Compile model with loss function and optimizer
model.compile(loss=CrossEntropyLoss(), optimizer=SGD(learning_rate=0.01))
# Train model on dummy data
X_train = np.random.rand(100, 4) # Random input data (100 samples, 4 features)
y_train = np.random.randint(0, 3, size=(100,)) # Random labels (3 classes) model.fit(X_train, y_train, epochs=10, batch_size=16)
model.fit(X_train, y_train, epochs=10, batch_size=16)
```
## Contributing

Contributions are welcome! If you’d like to contribute to PyNN, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
