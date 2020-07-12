#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:25:31 2020

@author: Thore
"""
from subprocess import call
import os
from pathlib import Path
import cv2
import csv
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
        
    
    def setup_data(self):
               
        test_fraction = 0.3
        cases = ['real', 'fake']
        files = []
        files.append( os.listdir('videos/' + cases[0]))
        files.append( os.listdir('videos/' + cases[1]))
        #For each case
        for i in range(2):
            # split data set into train and test
            num_files_test = round(len(files[i]) * test_fraction)
            j = -1
            for filename in files[i]:
                j+=1
                if j < num_files_test:
                   # it is a test file
                   train_or_test = 'test'
                else:
                    # it is a train file
                    train_or_test = 'train'
                    
                src = os.path.join('videos', cases[i], filename)
                filename_, ext = filename.split('.')
                folderdir = os.path.join('data', train_or_test, cases[i],
                                         filename_)
                # Path to chop video into sections
                Path(folderdir).mkdir(parents=True, exist_ok=True)
                dest = folderdir + '/chunk%03d.mp4'
                # Split video into sections
                call(["ffmpeg", "-i", src, "-c", "copy", "-map", "0", 
                     "-segment_time", "00:00:10",  "-f", "segment",  
                     "-reset_timestamps", "1", dest])
                chunks = os.listdir(folderdir)
                #Get size of video 
                cap = cv2.VideoCapture(folderdir + '/chunk000.mp4')
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                cap.release()
                #create folder of jpgs for each chunk
                for chunk in chunks:
                    chunkname, ext = chunk.split('.')
                    dest = folderdir + '/' + chunkname + '/frame%04d.jpg'
                    Path(folderdir + '/' + chunkname).mkdir(parents=True, exist_ok=True)
                    call(["ffmpeg", "-i", folderdir + '/' + chunk, dest])
                    # Remove chunk video
                    os.remove(folderdir + '/' + chunk)
                    # Count number of frames 
                    num_frames = len(os.listdir(folderdir + '/' + chunkname))
                    
                    self.data.append([train_or_test, cases[i], filename_, 
                                 chunkname, num_frames, (height, width)])
                    
        with open('data_file.csv', 'w') as fout:
            writer = csv.writer(fout)
            writer.writerows(self.data)
        print("Extracted and wrote %d video files." % (len(self.data)))
        
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
            
            #Get one-hotß
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
