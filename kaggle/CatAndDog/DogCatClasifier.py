import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import os, cv2, random


CHANNELS = 3

ROWS = 64
COLS = 64
JUPYTER = True

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

if JUPYTER:
    TRAIN_DIR = '../../../input/dog_cat/train/'
    TEST_DIR = '../../../input/dog_cat/test/'


TEST_RATIO = 0.5

# Image path array
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = train_images[:int(len(train_images) * TEST_RATIO)]
train_dogs = train_dogs[:int(len(train_dogs) * TEST_RATIO)]
train_cats = train_cats[:int(len(train_cats) * TEST_RATIO)]
test_images = test_images[:int(len(test_images) * TEST_RATIO)]



# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
train_images = train_dogs[:] + train_cats[:]
random.shuffle(train_images)
test_images =  test_images[:]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if (i % (count/100)) == 0: print('Processed {}% of {} complete'.format(i/(count/100), count), end='\r')
    
    return data

# Read image data from image path
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


nb_classes = 2
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
batch_size = 128
nb_epoch = 12


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(labels_train, nb_classes)
Y_test = np_utils.to_categorical(labels_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, 3, 3,
                        border_mode='valid',
                        input_shape=(64, 64, 3)))
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

train = train.astype('float32')
test = test.astype('float32')

train = train.reshape(train.shape[0], 64, 64, 3)
test = test.reshape(test.shape[0], 64, 64, 3)
print('train shape:', train.shape)
print('test shape:', test.shape)

print(train.shape, Y_train.shape, test.shape,Y_test.shape)
print((ROWS, COLS, CHANNELS))

model.fit(train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test, Y_test))

score = model.evaluate(test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])