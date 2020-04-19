import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10


def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = np.genfromtxt('/kaggle/input/digit-recognizer/test.csv', delimiter=',', skip_header=1, dtype=int)
X_test_orig = test[:, :]
X_test = X_test_orig / 255
X_final = X_test.reshape(28000, 28, 28, 1)

print(train.shape)
print(test.shape)

train_x, train_y = data_prep(train)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=128,
          epochs=5)

result = model.predict(X_final)
result = np.argmax(result, axis=1)
result = pd.Series(result, name="Label")
print(result)

submission_file = pd.concat([pd.Series(range(1, 28001), name="Id"), result], axis=1)
submission_file.to_csv("submission_file.csv", index=False)

