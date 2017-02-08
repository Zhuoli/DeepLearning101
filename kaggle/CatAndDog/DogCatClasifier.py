# https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
# https://www.kaggle.com/jeffd23/dogs-vs-cats-redux-kernels-edition/catdognet-keras-convnet-starter

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
#%matplotlib inline 

JUPYTER = True

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

if JUPYTER:
    TRAIN_DIR = '../../../input/dog_cat/train/'
    TEST_DIR = '../../../input/dog_cat/test/'

ROWS = 64
COLS = 64
CHANNELS = 3
nb_classes = 2
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
batch_size = 128
nb_classes = 10
nb_epoch = 12

# Image path array
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

random.shuffle(train_images)


# Reads and resizes images: 64*64 pixels
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

# Load images to 4-D matrix
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for idx, image_file in enumerate(images):
        image = read_image(image_file)
        data[idx] = image.T
        if idx%250 == 0: print('Processed {} of {}'.format(idx, count))
    
    return data

# Resize images and load them to 4-D matrix
train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

labels_train = []
dog = 0
cat = 0
for path in train_images:
    if 'dog' in path.split('/')[-1]:
        labels_train.append(1)
        dog+=1
    else:
        labels_train.append(0)
        cat+=1

print("Train DOG: " + str(dog))
print("Train CAT:" + str(cat))

labels_test = []
dog = 0
cat = 0
for path in test_images:
    if 'dog' in path.split('/')[-1]:
        labels_test.append(1)
        dog+=1
    else:
        labels_test.append(0)
        cat+=1

print("Test DOG: " + str(dog))
print("Test CAT:" + str(cat))


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(labels_train, nb_classes)
Y_test = np_utils.to_categorical(labels_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(ROWS, COLS, CHANNELS)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_images, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test_images, Y_test))
score = model.evaluate(test_images, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
