from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D

#Importing dataset
train_datagen = ImageDataGenerator(
                rescale=1./255,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
X_train = train_datagen.flow_from_directory(
                'datasets/train',
                target_size=(32,32),
                batch_size=32,
                class_mode='categorical')

test_datagen = ImageDataGenerator(
                rescale=1./255)
X_test = train_datagen.flow_from_directory(
                'datasets/test',
                target_size=(32,32),
                batch_size=32,
                class_mode='categorical')

#resnet model
base_model = ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(32,32,3), pooling=None, classes=10)
base_model.summary()
model = Sequential()
model.add(base_model)
model.add(GlobalMaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.7))                 
model.add(Dense(10, activation='softmax'))
model.summary()

#Training resnet model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, validation_data=X_test, epochs=5)

