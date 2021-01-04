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

#sklearn train-test-split
from sklearn.model_selection import train_test_split


powerproduction = pd.read_csv("powerproduction.csv", header = 0)
#powerproduction = powerproduction[~np.all(powerproduction[["power"]] == 0, axis=1)]

X_train, X_test, y_train, y_test = train_test_split(powerproduction[['speed']], powerproduction[['power']], test_size=0.2, random_state=42)
X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)

# Plot style.
plt.style.use("ggplot")

# Plot size.
plt.rcParams['figure.figsize'] = [14, 8]
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=16)

model = keras.models.Sequential()
model.add(layers.Dense(128, activation='relu',input_dim=1))
model.add(layers.LayerNormalization(axis=1))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
adam = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(X_train,y_train,epochs=1200, batch_size = 32,verbose = 1,callbacks=[callback],validation_data=[X_test,y_test],validation_split=0.2)


# Plot the predictions (on the training set itself).
plt.plot(powerproduction[['speed']], powerproduction[['power']], label='actual')
plt.plot(powerproduction[['speed']], model.predict(powerproduction[['speed']]), label='prediction')
plt.legend();

print("Value: 2.439 Prediction:", model.predict(np.array([1.702])))
model.save("myModel.h5")