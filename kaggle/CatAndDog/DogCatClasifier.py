import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
import os, cv2, random
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


CHANNELS = 3

ROWS = 64
COLS = 64
JUPYTER = True

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

if JUPYTER:
    TRAIN_DIR = '../../../input/dog_cat/train/'
    TEST_DIR = '../../../input/dog_cat/test/'


TEST_RATIO = 0.7

SAMPLE_SIZE_RATIO = 0.1

# Image path array
images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
dog_paths =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
dog_paths = dog_paths[:int(len(dog_paths)*SAMPLE_SIZE_RATIO)]

cat_paths =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
cat_paths = cat_paths[:int(len(cat_paths)*SAMPLE_SIZE_RATIO)]


train_dog_paths = dog_paths[:int(len(dog_paths) * TEST_RATIO)]
train_cat_paths = cat_paths[:int(len(cat_paths) * TEST_RATIO)]
# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
train_image_paths = train_dog_paths[:] + train_cat_paths[:]

# Pick the rest training data for validation
test_dog_paths = dog_paths[int(len(dog_paths) * TEST_RATIO) : ]
test_cat_paths = cat_paths[int(len(cat_paths) * TEST_RATIO) : ]
test_image_paths = test_dog_paths[:] + test_cat_paths[:]


random.shuffle(train_image_paths)
random.shuffle(test_image_paths)

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
train = prep_data(train_image_paths)
test = prep_data(test_image_paths)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))



labels_train = []
dog = 0
cat = 0
for path in train_image_paths:
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
for path in test_image_paths:
    if 'dog' in path.split('/')[-1]:
        labels_test.append(1)
        dog+=1
    else:
        labels_test.append(0)
        cat+=1

print("Test DOG: " + str(dog))
print("Test CAT:" + str(cat))

# number of convolutional filters to use
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
batch_size = 128
nb_epoch = 5

def show_cats_and_dogs(idx):
    cat = read_image(train_cat_paths[idx])
    dog = read_image(train_dog_paths[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def catdog():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = catdog()

#train = train.reshape(train.shape[0], 64, 64, 3)
#test = test.reshape(test.shape[0], 64, 64, 3)
print('train shape:', train.shape)
print('test shape:', test.shape)

print(train.shape, test.shape)
print((ROWS, COLS, CHANNELS))
## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   



history = LossHistory()
model.fit(train, labels_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_split=0.25,validation_data=(test, labels_test), verbose=1, shuffle=True, callbacks=[history, early_stopping])

score = model.evaluate(test, labels_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])