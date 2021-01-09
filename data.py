#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel DeepFake Challenge
@author: Thore
"""

from random import shuffle
import glob
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.utils import to_categorical

class DataSet():
    def __init__(self, read_index = True):
        if read_index:
            self.data = self.get_setup()
        else:
            self.data = []; 
            print('Either load data (get_data) or process videos (setup_data)')
        
    

        
    def get_setup(self):
        with open('data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
            
        return data
        
    def get_data(self, train_or_test, num_files = None, seq_length = 100, image_shape = (300, 300, 3)):
        
        out_list = [] #list of videos to parse      
        for item in self.data:
            if item[0] == train_or_test:
                out_list.append(item)

        if num_files is None:
            num_files = len(out_list)
        
        shuffle(out_list)
        print("loading %d out of %d samples" % (num_files, len(out_list)))
        out_list_ = out_list[0:num_files]
        x, y = [], [] 
        for row in out_list_:
            print(row)
            filename = os.path.join('data', row[0], row[1], row[2], row[3])
            #Load frame paths
            frames = sorted(glob.glob(filename+'/*.jpg'))
            #get desired number of frames 
            try:
                frames = frames[0:seq_length]
            except IndexError:
                print('warning, skipping file')
                continue #skip this frame, recorded sequence too short

            #Load frames into memory
            sequence = [load_image(x, image_shape) for x in frames]
            
            #Get one-hot
            if row[1] == 'real':
                label = to_categorical(0,2)
            elif row[1] == 'fake':
                label = to_categorical(1,2)
            else:
                print('label not identified')
            
            x.append(sequence)
            y.append(label)
            
        return np.array(x), np.array(y)
    

def load_image(image, target_shape):
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x
