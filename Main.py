from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from Dense import Layer_Dense


iris = load_iris()

df = pd.DataFrame(data = iris.data, columns=iris.feature_names)

df['targets'] = iris.target

X = df.drop('targets', axis=1).values
y = df['targets'].values


dense1  = Layer_Dense(4, 64)

dense2 = Layer_Dense(64, 3)

dense1.forward(X)

dense2.forward(dense1.output)

print(dense2.output)