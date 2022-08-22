# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils

img_size = 32
img_channels = 3
nb_classes = 10
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 5
images_per_file = 10000 
# number of all the images in the training dataset
nb_images_train = nb_files_train * images_per_file
epochs=25

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=256,
                 kernel_size=(2, 2),
                 kernel_initializer='he_normal',
                 activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

history=model.fit(x_train, y_train,
          batch_size=32,
          epochs=18,
          verbose=1,
          validation_data=(x_test, y_test))
import numpy as np
loss, accuracy = model.evaluate(x_test, y_test)
result = model.predict(x_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
accuracy = float(num_correct)/result.shape[0]

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()