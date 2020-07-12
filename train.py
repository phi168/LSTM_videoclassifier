#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:44:53 2020

@author: Thore
"""

from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from data import DataSet

#%% Get data
seq_length = 100
num_files = 3
image_shape = (256+6, 256+6, 3)

d = DataSet()
x, y = d.get_data('train', num_files = 3, seq_length = 10,
                  image_shape = image_shape)
x_test, y_test = d.get_data('test', num_files = 3, seq_length = 10,
                  image_shape = image_shape)

#%% Build Model
batch_size = num_files #batch gradient decent
num_epochs = 100
num_features = 2048

# vgg_model = VGG16(input_shape = image_shape, 
#                   weights = 'imagenet', include_top = False)

# cnn = Model(vgg_model, input_shape = (seq_length,) + image_shape)

model = Sequential()
#might want to use dilation here
model.add(TimeDistributed(Conv2D(filters = 16, kernel_size = (7,7), 
               strides = (2,2), padding = 'valid', activation = 'relu'), 
                          input_shape = x.shape[1:], ))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 32, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 64, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 128, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 256, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 512, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Conv2D(filters = 1024, kernel_size = (3,3), 
               padding = 'same', activation = 'relu')))
model.add(TimeDistributed(MaxPool2D(pool_size = (2,2), strides = (2,2))))
model.add(TimeDistributed(Dense(units = num_features, activation = 'relu')))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(32, return_sequences=False,
               dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

print(model.summary())
#%% Fit
model.fit(x,y, 
          batch_size = batch_size, 
          validation_data = (x_test, y_test), 
          epochs = num_epochs,
          verbose = 1)
