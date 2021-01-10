
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use MTCNN to extract face locations in video
@author: Thore
"""
import glob
import cv2
import mtcnn
import pickle
import os
from tqdm import tqdm
#%% use MTCNN to extract position of faces in videos
root = 'resources/'
files = glob.glob(os.path.join(root + 'videos', 'real','*.mp4'))
detector = mtcnn.MTCNN()

for j in tqdm(range(len(files))): #for all videos
    #get file path to video
    file = files[j] 
    #skip if file has already been processed
    if os.path.isfile(file+'faces.p'):
        continue
    
    cap = cv2.VideoCapture(file)#open video
    #num frames in video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    faces = []
    i = 0
    while cap.isOpened():
        #lopp through frames
        ret, frame = cap.read()
        i += 1
        if ret==True:   #managet to get next frame
            #get faces
            f = detector.detect_faces(frame)
            faces.append([x['box'] for x in f])
        else:
            break
    cap.release()
    #print('writing to ' + file + 'faces.p')
    pickle.dump(faces, open(file+'faces.p', 'wb'))