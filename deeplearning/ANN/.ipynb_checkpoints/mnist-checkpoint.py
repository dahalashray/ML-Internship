import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

#Importing the dataset
train_ds = pd.read_csv('mnist_train.csv')
X_train = train_ds.iloc[:, 1:].values
y_train = train_ds.iloc[:, 0].values
test_ds = pd.read_csv('mnist_test.csv')
X_test = train_ds.iloc[:, 1:].values
y_test = train_ds.iloc[:, 0].values

#Spiltting the dataset into train and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

#Encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#model
classifier = Sequential()
classifier.add(Dense(units=32, activation='sigmoid', input_dim=784))
classifier.add(Dense(units=64, activation='sigmoid'))
classifier.add(Dense(units=128, activation='sigmoid'))
classifier.add(Dense(units=10, activation='softmax'))
print(classifier.summary())

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.optimizer.lr=0.001

#training the model
classifier.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=100)

#Prediction
y_pred = classifier.predict(X_test)
y_pred_cm = np.argmax(y_pred, axis=0)
y_test_cm = np.argmax(y_pred, axis=0)

cm = confusion_matrix(y_test_cm,y_pred_cm)
print(cm)