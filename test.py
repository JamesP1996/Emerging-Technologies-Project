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

import random

percent = 0.45
powerproduction = pd.read_csv("powerproduction.csv", header = 0 , skiprows = lambda i: i>0 and random.random() > percent)
powerproduction = powerproduction[~np.all(powerproduction[["power"]] == 0, axis=1)]
print(powerproduction)

# Plot style.
plt.style.use("ggplot")

# Plot size.
plt.rcParams['figure.figsize'] = [14, 8]


# Train a different model.
#model = keras.models.Sequential()
#model.add(layers.BatchNormalization(axis=1))
#model.add(layers.Dense(50,input_shape=(1,), activation='sigmoid'))
#model.add(layers.Dense(50, activation='sigmoid'))
#model.add(layers.Dense(1, activation='linear'))
#model.compile(keras.optimizers.Adam(lr=0.01 ,decay=1e-6),loss="mean_squared_error",metrics=["mean_absolute_error"])


# Fit the data.
#model.fit(powerproduction['speed'], powerproduction['power'], epochs=1000)

model = keras.models.Sequential()
model.add(layers.Dense(100, activation='relu', 
                input_dim=1))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
adam = keras.optimizers.Adam(lr=0.01, decay=1e-6)
model.compile(loss='mean_squared_error', 
              optimizer=adam,
              metrics=['accuracy'])
model.fit(powerproduction['speed'],powerproduction['power'],epochs=1000, batch_size = 32,verbose = 0)


# Plot the predictions (on the training set itself).
plt.plot(powerproduction[['speed']], powerproduction[['power']], label='actual')
plt.plot(powerproduction[['speed']], model.predict(powerproduction[['speed']]), label='prediction')
plt.legend();

print("Value: 2.439 Prediction:", model.predict(np.array([1.702])))
model.save("myModel.h5")