from __future__ import print_function

import keras.callbacks as cb
from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout

#  The Sequential model or the Model class used with functional API.
from keras.models import Sequential
# Stochastic gradient descen
from keras.optimizers import SGD
# Norm regularizers
from keras.regularizers import l1, l2
from keras.utils import np_utils

from matplotlib import pyplot as plt
import numpy as np
import time


# Min-Max Normalize value to [0, 1]
def PreprocessDataset(loadMethod, categoriesNum):
    # from sklearn import preprocessing
    ## Load MNIST dataset of handwritten digits
    (x_train, y_train), (x_test, y_test) = loadMethod()
    ## Transform labels to one-hot encoding
    ## i.e., from '7' to [0,0,0,0,0,0,0,1,0,0]
    y_train = np_utils.to_categorical(y_train, categoriesNum)
    y_test = np_utils.to_categorical(y_test, categoriesNum)

    ## Process features. Set numeric type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    ## X_train has 6000 samples with 28x28 features
    ## Reshape from a matrix of 28 x 28 pixels to 1-D vector of 784 dimensions
    xtrain_shape = x_train.shape
    xtest_shape = x_test.shape
    if (len(xtrain_shape) == 3):
        xtrain_shape = (xtrain_shape[0], xtrain_shape[1], xtrain_shape[2], 1)
    if (len(xtest_shape) == 3):
        xtest_shape = (xtest_shape[0], xtest_shape[1], xtest_shape[2], 1)
    x_train = np.reshape(x_train, (xtrain_shape[0], xtrain_shape[1] * xtrain_shape[2] * xtrain_shape[3]))
    x_test = np.reshape(x_test, (xtest_shape[0], xtest_shape[1] * xtest_shape[2] * xtest_shape[3]))

    ################################################################
    # Activity 1 (Pre-processing):
    # Group A: w/o pre-processing datasets.
    #
    # Group B: Min-Max Normalize value to [0, 1]
    # x_train /= 255
    # x_test /= 255
    #
    # Group C: proceed w/ standardizing datasets by z-scoring (de-mean, uni-variance).
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)
    ################################################################
    ## YOUR TURN: CHANGE HERE
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


# Two layer with shrink width network
# Using Relu
# Using crossentropy
# Without Dropput
def DefineModel(input_dim):
    first_layer_width = 128
    second_layer_width = 64
    third_layer_width = 0

    ################################################################
    # Activity 3 (Activation Function):
    # Group A uses ReLU.
    # activation_func = 'relu'
    #
    # Group B uses Sigmoid.
    # activation_func = 'sigmoid'
    #
    # Group C uses Tanh.
    # activation_func = 'tanh'
    ################################################################
    activation_func = 'relu'

    ################################################################
    # Activity 4 (Loss Function):
    # Group A uses cross entropy.
    # loss_function = 'categorical_crossentropy'
    #
    # Group B uses cross entropy.
    # loss_function = 'categorical_crossentropy'
    #
    # Group C uses squared error.
    # loss_function = 'mean_squared_error'
    ################################################################
    loss_function = 'categorical_crossentropy'
    dropout_rate = 0.0

    ################################################################
    # Activity 6 (Regularization):
    # Group A uses L1 regularizer
    # weight_regularizer = l1(0.01)
    #
    # Group B uses L2 regularizer
    # weight_regularizer = l2(0.01)
    #
    # Group C uses no regularizer
    # weight_regularizer = None
    ################################################################
    weight_regularizer = None

    learning_rate = 0.1

    ## Initialize model.
    model = Sequential()

    ## First hidden layer with 'first_layer_width' neurons.
    ## Also need to specify input dimension.
    ## 'Dense' means fully-connected.
    model.add(Dense(first_layer_width, input_dim=input_dim, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    if dropout_rate > 0:
        model.add(Dropout(0.5))

    ## Second hidden layer.
    if second_layer_width > 0:
        model.add(Dense(second_layer_width))
        model.add(Activation(activation_func))
        if dropout_rate > 0:
            model.add(Dropout(0.5))

    if third_layer_width > 0:
        model.add(Dense(third_layer_width))
        model.add(Activation(activation_func))

    ## Last layer has the same dimension as the number of classes
    model.add(Dense(NumOfCategories))
    ## For classification, the activation is softmax
    model.add(Activation('softmax'))
    ## Define optimizer. In this tutorial/codelab, we select SGD.
    ## You can also use other methods, e.g., opt = RMSprop()
    opt = SGD(lr=learning_rate, clipnorm=5.)
    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])

    return model


def TrainModel(data=None, epochs=20):
    ################################################################
    # Activity 7 (Mini-batch):
    # Group A uses mini-batch of size 128.
    # batch = 128
    #
    # Group B uses mini-batch of size 256.
    # batch = 256
    #
    # Group C uses mini-batch of size 512.
    # batch = 512
    ################################################################
    batch = 128
    start_time = time.time()
    model = DefineModel(data[0].shape[1])
    if data is None:
        print("Must provide data.")
        return
    x_train, x_test, y_train, y_test = data
    print('Start training.')
    ## Use the first 55,000 (out of 60,000) samples to train, last 5,500 samples to validate.
    history = model.fit(x_train[:55000], y_train[:55000], nb_epoch=epochs, batch_size=batch,
                        validation_data=(x_train[55000:], y_train[55000:]))
    print("Training took {0} seconds.".format(time.time() - start_time))
    return model, history


def TestModel(model=None, data=None):
    if model is None:
        print("Must provide a trained model.")
        return
    if data is None:
        print("Must provide data.")
        return
    x_test, y_test = data
    scores = model.evaluate(x_test, y_test)
    return scores


## DRAW BELOW

def PlotHistory(train_value, test_value, value_is_loss_or_acc):
    f, ax = plt.subplots()
    ax.plot([None] + train_value, 'o-')
    ax.plot([None] + test_value, 'x-')
    if "Accuracy" in value_is_loss_or_acc:
        ax.set_ylim(0, 1)
    ## Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train ' + value_is_loss_or_acc, 'Validation ' + value_is_loss_or_acc], loc=0)
    ax.set_title('Training/Validation ' + value_is_loss_or_acc + ' per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value_is_loss_or_acc)
    plt.show()


def drawWeightHistogram(x):
    ## the histogram of the data
    fig = plt.subplots()
    n, bins, patches = plt.hist(x, 50)
    plt.xlim(-0.5, 0.5)
    plt.xlabel('Weight')
    plt.ylabel('Count')
    zero_counts = (x == 0.0).sum()
    plt.title("Weight Histogram. Num of '0's: %d" % zero_counts)
    plt.show()


def ShowInputImage(data, datashape):
    """Visualize input image."""
    plot = plt.figure()
    plot.set_size_inches(2, 2)
    plt.imshow(np.reshape(-data, datashape), cmap='Greys_r')
    plt.title("Input Image")
    plt.axis('off')
    plt.show()


def ShowHiddenLayerOutput(trained_model, input_data, target_layer_num):
    """Visualize output from the target hidden layer."""
    from keras import backend as K
    ## Backend converter: to TensorFlow
    target_layer = K.function(trained_model.inputs, [trained_model.layers[target_layer_num].output])
    ## Extract output from the target hidden layer.
    target_layer_out = target_layer([input_data])
    plot = plt.figure()
    plot.set_size_inches(2, 2)
    plt.imshow(np.reshape(-target_layer_out[0][0], (16, -1)), cmap='Greys_r')
    plt.title("Hidden layer " + str(target_layer_num))
    plt.axis('off')
    plt.show()


def ShowFinalOutput(trained_model, input_data):
    """Calculate final prediction."""
    from keras import backend as K
    ## Backend converter: to TensorFlow
    ## Calculate final prediction.
    last_layer = K.function(trained_model.inputs, [trained_model.layers[-1].output])
    last_layer_out = last_layer([input_data])
    print("Final prediction: " + str(np.argmax(last_layer_out[0][0])))


NumOfCategories = 10;

x_train, x_test, y_train, y_test = PreprocessDataset(mnist.load_data, NumOfCategories)
print("x_train type: " + str(x_train.shape))
print("x_test type: " + str(x_test.shape))
print("y_train type: " + str(y_train.shape))
print("y_test type: " + str(y_test.shape))

#
trained_model, training_history = TrainModel(data=[x_train, x_test, y_train, y_test], epochs=20)
#
#
#
test_score = TestModel(model=trained_model, data=[x_test, y_test])
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
#
#
PlotHistory(training_history.history['loss'], training_history.history['val_loss'], 'Loss')
PlotHistory(training_history.history['acc'], training_history.history['val_acc'], 'Accuracy')
print(y_test.shape[0])
print(y_test[0])

X = 0
groupX = []
for idx in range(0, y_test.shape[0]):
    if y_test[idx][X] == 1.0:
        groupX.append(idx)

if (len(groupX) == 0):
    print("No images found in group: " + str(X))

for idx in groupX[:5]:
    ShowInputImage(x_test[idx], (28,28))


# w1 = trained_model.layers[0].get_weights()[0].flatten()
# drawWeightHistogram(w1)
# (x_test[0])
# ShowHiddenLayerOutput(x_test, 1)
# ShowFinalOutput(x_test)
