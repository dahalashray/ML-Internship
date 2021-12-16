import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization, Dropout

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
y_training = to_categorical(y_training,10)
y_val = to_categorical(y_val,10)

#alexnet network
model = Sequential()
#add model layers

#Conv 1
model.add(Conv2D(filters=96, input_shape=(28,28,1), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

#Conv 2
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
# Batch Normalisation
model.add(BatchNormalization())

#Conv 3
model.add(Conv2D(filters=384,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPool2D(pool_size=(3,3),  strides= (2,2),padding='same'))
#Normalization
model.add(BatchNormalization(axis=-1))

#Conv4
# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
# Pooling 
model.add(MaxPool2D(pool_size=(2,2),  strides= (2,2),padding='same'))
#Normalization
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096,input_shape=(28*28*1,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_t, y_training,validation_data=(X_v,y_val),epochs=20,batch_size=64)