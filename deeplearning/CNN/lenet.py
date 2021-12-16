import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

#import dataset
train = pd.read_csv('datasets/mnist_train.csv')
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_train = np.array(X_train)
y_train = np.array(y_train)

#spilting training and validation sets
X_training, X_val, y_training, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0, shuffle=True)
#reshaping and normalising
X_t = X_training.reshape(X_training.shape[0], 28, 28, 1)
X_v = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_t=X_t/255
X_v=X_v/255
#encoding
y_training = to_categorical(y_training)
y_val = to_categorical(y_val)

#Lenet network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(16,16),strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_t, y_training, validation_data=(X_v,y_val), epochs=25)

#preprocessing X_test for prediction
img1 = X_train[1]
img1 = img1.reshape(28,28,1)
img2 = X_train[10000]
img2 = img2.reshape(28,28,1)

#Batch size
img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)
img_vs = np.vstack([img1,img2])
img_vs.shape

#Prediction
y_pred = model.predict(img_vs)
y_pred.shape
for i in y_pred:
    print(np.argmax(i))