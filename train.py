#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:44:53 2020

@author: Thore
"""

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger
from data import DataSet
import os
import time
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
num_epochs = 10
num_features = 2048

# vgg_model = VGG16(input_shape = image_shape, 
#                   weights = 'imagenet', include_top = False)

# cnn = Model(vgg_model, input_shape = (seq_length,) + image_shape)

model = Sequential()

#Generic CNN to decrease dimensions down to num_features
#Something deeper might be necessary
#might want to use dilation in first 1-2 layers to allow for higher-res images
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
# Long Short Term Memory to identify temporal featurs 
# E.g. face moving weirdly in relation to rest of body
model.add(LSTM(32, return_sequences=False,
               dropout=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

print(model.summary())
#%% Fit
# Callbacks 
tb = TensorBoard(log_dir=os.path.join('data', 'logs'))
early_stopper = EarlyStopping(patience=5)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'training-' + \
        str(timestamp) + '.log'))

model.fit(x,y, 
          batch_size = batch_size, 
          validation_data = (x_test, y_test), 
          epochs = num_epochs,
          verbose = 1, 
          callbacks = [tb, early_stopper, csv_logger])

#%%
model.save(os.path.join('model','model.keras'))