# -*- coding: utf-8 -*-
# Objective: Predict the upward or downward trend of Google Stock Price (predict 'Open' (as dependent) variable)

# Part-1: Data Preprocessing
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = (dataset_train).iloc[:, 1:2].values

# Feature Scaling -> We have a sigmoid funtion in the output layer of our neural network so we apply normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output -> much like k-fold
X_train = []
y_train = []

for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0]) # time(60): t_(0-60)
    y_train.append(training_set_scaled[i, 0]) # time(61): t_(61)
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part-2: Building the RNN
# impoting the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some dropout regularisation
# units- the higher the number of neurons the better(your hardware is the limiting factor) also too high may cause overfitting
# return_sequence- the inner layers will have this parameter set to True but the last outer layer to False.
# this is because we don't need the last layer to return the sequence. We simply make our final predictions from all the updated weights
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(rate = 0.2))

# Adding the second LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the third LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the fourth (final) LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss='mean_squared_error')# 'rmsprop' is us usually a good choice for RNNs

# Fitting the RNN to the Training set- connecting to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part-3: Making the predictions and visualising the results.
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = (dataset_test).iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)

# Scaling the inputs
inputs = sc.transform(inputs)

# Instantiating the special data structure for our test set
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0]) # time(60): t_(0-60)
X_test= np.array(X_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting the results
predicted_stock_price = regressor.predict(X_test)

# Inverse scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()







