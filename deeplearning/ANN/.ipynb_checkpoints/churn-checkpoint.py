import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Preprocessing
# Male/Female
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

#Spiltting the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Scaling the dataset
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)

#model
classifier = Sequential()
classifier.add(Dense(units=6, activation='relu', input_dim=11))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#compiling the ann
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training ann
classifier.summary()
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
classifier.fit(X_train, y_train, batch_size=32, epochs=100)

#testing
X_test = X_test.astype('float32')
y_pred = classifier.predict(X_test)

print(y_pred)