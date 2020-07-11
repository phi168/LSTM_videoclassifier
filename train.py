#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:44:53 2020

@author: Thore
"""

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from data import DatSet

#%% Get data
data = DataSet()
x, y = data.get_data('train')
x_test, y_test = data.get_data('test')

#%% Build Model
model = 'lstm'
batch_size = 32
nb_epoch = 1000
data_type = 'features'
seq_length = 100
feature_length = 100
input_shape = (seq_length, feature_length)
num_classes = 2


model = Sequential()
model.add(LSTM(2048, return_sequences=False,
               input_shape=input_shape,
               dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

#%% Fit
model.fit(x,y, 
          batch_size = batch_size, 
          validation_data = (x_test, y_test), 
          epoch = num_epochs,
          verbose = 1)
