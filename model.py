import tensorflow as tf


from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D

from scipy.misc import imread, imsave, imresize 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import cv2 

#################################################################
# By Vivek Yadav :
#https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
#################################################################
#transformation of  the image
def trans_image(image,steering):

    
    rows, cols, _ = image.shape
   
    cv2.imwrite("jitter1.jpg",image)
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    cv2.imwrite("jitter2.jpg",image)
    # Convert the image into mulitdimensional matrix of float values (normally int which messes up our division).
    image = np.array(image, np.float32)

    
    return image,steering


# flipp the image
def flipped(image, measurement):
  return np.fliplr(image), -measurement

# resize the image not used in this pipeline
def resize(image, new_dim):
  resized_image = cv2.resize(image, (200, 66)) 
  return resized_image

#takining a images from drive and send to processing
def get_image(i, data):
  #resize_dim=(66, 200)
  PATH = "simulator_data/IMG/"
  positions, corrections = ['left', 'center', 'right'], [.25, 0, -.25]
  ID, r = data.index[i], random.choice([0, 1, 2])
  measurement = data['steering'][ID] + corrections[r]

  path = data[positions[r]][ID][0:]
  path = path.split('/')[-1]
  #path = path.split('C:\\Python35\\simulator_data\\IMG\\')[-1]

  path = PATH + path
  image = imread(path)
  

  if r == 1:
    if random.random() > 0.5:
      image = imread(path)
      
    else: image, measurement  = trans_image(image,measurement)
   
  if random.random() > 0.5:
    image, measurement = flipped(image, measurement)
 
  return image, measurement


#generator to help with memory management
def generator(data, batch_size):

    num_samples = len(data)

    while 1:
        
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            for batch_sample in range(offset, offset + batch_size):
                if batch_sample < num_samples:
                    image, angle = get_image(batch_sample, data)
                    angles.append(angle)
                    #image = resize(image, (66, 200))
                    images.append(image)

            yield np.array(images), np.array(angles)

############ MODEL ###################

# Comma.ai model
# https://github.com/commaai/research/blob/master/train_steering_model.py

learning_rate= 0.0001

model = Sequential()

model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(ELU())
model.add(Dense(1))

model.summary()
model.compile (optimizer=Adam(learning_rate), loss = "mse", metrics=['accuracy'])

############ END OF MODEL ###################


batch_size = 32
number_of_epochs = 5
PATH = "simulator_data/IMG/"
csv_file = "C:/Python35/simulator_data/driving_log.csv"


DATA = pd.read_csv(csv_file, usecols = [0, 1, 2, 3])


training_data, validation_data = train_test_split(DATA, test_size = 0.15)
length_training_data = len(training_data)
length_validation_data = len(validation_data)

#################################################################

print('Training model...')

training_generator = generator(training_data, batch_size = batch_size)
validation_generator = generator(validation_data, batch_size = batch_size)


history = model.fit_generator(training_generator,
                 samples_per_epoch = length_training_data,
                 validation_data = validation_generator,
                 nb_val_samples = length_validation_data,
                 nb_epoch = number_of_epochs,
                 verbose = 2)

print(history.history.keys())

### plot the training and validation loss for each epoch

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('MSE.png', bbox_inches='tight')
#plt.show()

#################################################################
#saving a model
print('##################')
print('##################')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("CCN model saved to file.")


