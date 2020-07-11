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

class DataSet():
    def __init__(self):
        self.data = []; 
        
    
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
        print("Extracted and wrote %d video files." % (len(data_file)))
