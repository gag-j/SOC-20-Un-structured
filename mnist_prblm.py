import tensorflow as tf
import keras
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

batch_size = 64
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
model.add(BatchNormalization(axis = -1, momentum = 0.999, epsilon = 0.0001))
model.add(Dropout(0.25))
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization(axis = -1, momentum = 0.999, epsilon = 0.0001))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False), metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = epochs, verbose = 1, validation_data = (x_test, y_test))

history.history['accuracy']

score = model.evaluate(x_test, y_test, verbose = 0)

print('test loss :', score[0])
print('test accuracy :', score[1])

