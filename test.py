# Data Frame Imports.
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Numpy Import
import numpy as np

# Machine Learning Imports (Keras)
import tensorflow as tf 
from tensorflow import  keras
from tensorflow.keras import layers


# Simple linear equation.
f = lambda x: 2.0 * x**2 + 3.0 * x + 4.0

powerproduction = pd.read_csv("powerproduction.csv")
print(powerproduction)

# Plot style.
plt.style.use("ggplot")

# Plot size.
plt.rcParams['figure.figsize'] = [14, 8]

# Create a new neural network.


# Train a different model.
model = keras.models.Sequential()
model.add(layers.Dense(25, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.add(layers.Dense(25, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.add(layers.Dense(25, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.add(layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.compile(keras.optimizers.Adam(lr=0.001), loss='mean_squared_error')


# Fit the data.
model.fit(powerproduction['speed'], powerproduction['power'], epochs=1024, batch_size=32)


# Plot the predictions (on the training set itself).
plt.plot(powerproduction[['speed']], powerproduction[['power']], label='actual')
plt.plot(powerproduction[['speed']], model.predict(powerproduction[['speed']]), label='prediction')
plt.legend();

model.predict(np.array([1, 2,]))

# Add a single neuron in a single layer, initialised with weight 1 and bias 0.
#model.add(layers.Dense(1, input_dim=1, activation="sigmoid", kernel_initializer=keras.initializers.Constant(value=1.0), bias_initializer=keras.initializers.Constant(value=0.0)))

# Compile the model.
#model.compile(loss="mean_squared_error", optimizer="sgd")

# Train the neural network on our training data.
#model.fit(powerproduction[['speed']], powerproduction[['power']], epochs=12)

# Create some input values.
#x = np.array(powerproduction[["speed"]])

# Run each x value through the neural network.
#y = model.predict(x)

# Plot the values.
#plt.plot(x, y, 'k.')