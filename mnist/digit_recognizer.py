import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train.label
X_train = train.drop("label", axis=1)
del train

X_train = X_train/255.
test = test/255.

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)

def ConvModel(input_shape):

    X_input = Input(input_shape)
    X = Conv2D(32, (5, 5), strides = (1, 1),padding = 'same', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (2, 2),strides = (2, 2), name = 'max_pool1')(X)
    X = Conv2D(64, (5, 5), strides = (1, 1), padding = 'same', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (2, 2),strides = (2, 2), name = 'max_pool2')(X)
    X = Conv2D(128, (5, 5), strides = (1, 1), padding = 'same', name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (2, 2), strides = (1, 1), name = 'max_pool3')(X)
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    X = Dense(10, activation = 'softmax', name = 'fc')(X)
    model = Model(inputs = X_input, outputs = X, name = 'HappyModel')
    
    return model

convModel = ConvModel(X_train.shape[1:])

convModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
convModel.fit(x = X_train, y = Y_train, epochs = 20 , batch_size = 128)

results = happyModel.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("my_submission.csv",index=False)

convModel.summary();
