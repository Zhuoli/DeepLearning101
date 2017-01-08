from TitanicPredictionKeras import *

# https://www.kaggle.com/l3nnys/titanic/dense-highway-neural-network-for-titanic
import pandas as pd
import numpy as np
from keras.utils import np_utils

# Reading data
# Train data
traindata = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

# Test data
testdata  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(traindata.head())
# print(testdata.shape)

traindata, testdata = cleanData(traindata, testdata)

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(traindata.head())
# print(testdata.shape)


# Defining columns to use in the model
columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt', 'Fare',
           'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived',
           'Female_Friends_NotSurvived',
           'MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']

# train/test split
x_train, y_train, x_test, y_test, _ = split_data(traindata[columns].values, traindata['Survived'].values, 0.2)

## Transform labels to one-hot encoding
## i.e., from '7' to [0,0,0,0,0,0,0,1,0,0]
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

scaled_x_train, scaled_x_test, scaled_test = standardScalerData(x_train=x_train, x_test=x_test, test=testdata, columns=columns)

# After data scalling
print("x train shape:   " + str(scaled_x_train.shape))
print("x test shape:    " + str(scaled_x_test.shape))
print("test shape:  " + str(scaled_test.shape))
print(y_train[:5])

from keras.models import Sequential
model = Sequential()
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#
# First hidden layer with 'first_layer_width' neurons.
# Also need to specify input dimension.
# 'Dense' means fully-connected.
dropout_rate = 0
second_layer_width=128
learning_rate = 0.1
model.add(Dense(128, input_dim=scaled_x_train.shape[1], W_regularizer=None))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

## Second hidden layer.
model.add(Dense(second_layer_width))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

model.add(Dense(second_layer_width))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1]))
## For classification, the activation is softmax
model.add(Activation('softmax'))
## Define optimizer. In this tutorial/codelab, we select SGD.
## You can also use other methods, e.g., opt = RMSprop()
opt = SGD(lr=learning_rate, clipnorm=5.)
## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(scaled_x_train, y_train, nb_epoch=20, batch_size=32)

# Final test predict
test_proba = model.predict(scaled_test)
print(test_proba.shape)
print(test_proba[:5])
test_classes = np_utils.probas_to_classes(test_proba)
print(test_classes.shape)
print(test_classes[:5])