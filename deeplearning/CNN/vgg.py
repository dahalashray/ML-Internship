import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Dropout, BatchNormalization

#importing and preprocessing training dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
X_train = train_datagen.flow_from_directory(
        'datasets/training',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')

#importing and preprocessing test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
X_test = test_datagen.flow_from_directory(
        'datasets/validation',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')

#VGG network

model = Sequential()
#Conv 1
model.add(Conv2D(filters=64, input_shape=(224,224,3), kernel_size=(3,3),\
 strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, input_shape=(224,224,3), kernel_size=(3,3),\
 strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
#Conv 2
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
#Conv 3
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
#Conv4
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
#Conv5
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(1,1), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.summary()

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization(axis=-1))

model.add(Dense(11,activation='softmax'))
          
model.summary()

#Training VGG
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
model.fit(x= X_train, validation_data= X_test, epochs= 10)