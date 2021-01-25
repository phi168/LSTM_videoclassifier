#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class handles the data
@author: Thore
"""

from random import shuffle
import glob
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.utils import to_categorical
import csv
import os

class DataSet():
    def __init__(self, filepath, datapath):
        """filepath: path to list of video folders, datapath: path to videos"""
        self.data = self._get_setup(filepath)
        self.root = datapath
        
    def _get_setup(self, filepath):
        """read file containing details about video locations"""
        
        with open(filepath, 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data
        
    def get_data(self, train_or_test, num_files = None, seq_length = 100, image_shape = (300, 300, 3)):
        """
        

        Parameters
        ----------
        train_or_test : STR
            'train' or 'test' data set.
        num_files : INT, optional
            How many files to return. The default is None.
        seq_length : int, optional
            number of frames per video. The default is 100.
        image_shape : tuple, optional
            target dimensions of each image. The default is (300, 300, 3).

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        out_list = [] #list of videos to parse      
        for item in self.data:
            if item[0] == train_or_test:
                out_list.append(item)

        if num_files is None:
            flag_load_all_files = True
            num_files = len(out_list)
        else:
            flag_load_all_files = False
            
        #ranomise order of list
        shuffle(out_list)
        
        print("loading %d out of %d samples" % (num_files, len(out_list)))
        
        x, y = [], [] 
        i = 0
        while len(x) < num_files:
            row = out_list[i]
            i += 1
            print(row)
            filename = os.path.join(d.root, 'data', row[0], row[1], row[2], row[3])
            #Load frame paths
            frames = sorted(glob.glob(filename+'/*.png'))
            
            #get desired number of frames                 
            if len(frames) < seq_length:
                if flag_load_all_files: 
                    num_files -= 1
                print('warning, skipping file')
                continue #skip this frame, recorded sequence too short
            frames = frames[0:seq_length]
            
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
    """load image and resize to target height and width"""
    
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x
