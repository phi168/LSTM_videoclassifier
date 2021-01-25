#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thore
"""

"""
Script to train 
"""
from keras.models import Sequential
from keras import Input
from keras.applications import ResNet50V2
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger
from .data import DataSet
import os
import time
from pathlib import Path
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(num_epochs = 100, 
          num_files = 'all', 
          seq_length = 50, 
          image_shape = (256+6, 256+6, 3), 
          filepath = 'data_file.csv', 
          datapath = 'resources/'):
    
    d = DataSet(filepath, datapath)
    if num_files == 'all':
        num_files_test = sum(y[0] == 'test' for y in d.data)
        num_files_train = sum(y[0] == 'train' for y in d.data)
    else:
        num_files_test = num_files
        num_files_train = num_files
        
    x, y = d.get_data('train', num_files = num_files_train, seq_length = seq_length,
                      image_shape = image_shape)
    x_test, y_test = d.get_data('test', num_files = num_files_test, seq_length = seq_length,
                      image_shape = image_shape)
    
    #% Build Model
    batch_size = num_files #batch gradient decent
    num_epochs = 100
    num_features = 2048

    resnet = ResNet50V2(include_top = False, input_shape = image_shape)
    model = Sequential()
    model.add(TimeDistributed(resnet, input_shape = (seq_length,) + image_shape))
    model.add(TimeDistributed(MaxPool2D(pool_size=(9, 9))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(units = num_features, activation = 'relu')))
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
    #% Fit
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
    
    #%
    Path('model').mkdir(parents=True, exist_ok=True)
    model.save(os.path.join('model','model.keras'))