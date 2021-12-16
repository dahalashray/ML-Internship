import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [1,2,3,4]].values
y = dataset.iloc[:, 5].values

#Encoding
le = LabelEncoder()
y = le.fit_transform(y)

#Spiltting the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle=True)

#Scaling the dataset
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)
y_train=utils.to_categorical(y_train,num_classes=3)
y_test=utils.to_categorical(y_test,num_classes=3)

#model
classifier = Sequential()
classifier.add(Dense(units=2, activation='relu', input_dim=4))
classifier.add(Dense(units=4, activation='relu'))
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training the model
classifier.fit(X_train, y_train, batch_size=30, epochs=400)

#prediction
y_pred = classifier.predict(X_test)

for i,j in zip(y_pred,y_test):
    print(np.argmax(j),np.argmax(i))

print(y_test)