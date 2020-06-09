# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:09:21 2020

@author: Coderash
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution 
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate = 0.1))

# Adding a second convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate = 0.1))

#  Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dense(units=3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Training',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Testing',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=1000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=500)

classifier.save('RGB_Classifier.h5')

import keras
classifier = keras.models.load_model('RGB_Classifier.h5')


import numpy as np
from keras.preprocessing import image

#Enter the file name instead of file_name

test_image = image.load_img("file_name",target_size=(64, 64) ) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0) 
result = classifier.predict(test_image)
#training_set.class_indices

if (result[0][0]==1):
    prediction='Blue'
elif (result[0][1]==1):
    prediction='Green'
else:
    prediction='Red'
print(prediction)
