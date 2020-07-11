
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel DeepFake Challenge
@author: Thore
"""
import glob
import cv2
import numpy as np
from scipy import ndimage 
 
# =============================================================================
# The idea is to remove information in the video which is not relevant to 
# it being fake or note. That is: All background (also ignore sound). 
# Fake videos tend to have more watermarks. Would be good to remove that as a
# feature tha algorithm trains on. 
# This didn't work very well using the openCV classifier, but there is lots of
# room for improvement. 
# - I will stick with unprocessed data for now
# - Algorithm can be switched out to a NN solution. 
# -- Would be ironic if the NN is so good that it doesn't identify fake faces
# -- But that is a low risk. Just use a less accurate NN. 
# =============================================================================

filenames =  glob.glob('videos/fake/*.mp4') + glob.glob('videos/real/*.mp4')
for j in range(len(filenames)):
    file = filenames[j]
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #%% Extract Faces
    backSub = cv2.createBackgroundSubtractorMOG2(100, 16, False) 
    cap = cv2.VideoCapture(file)
    
    faces_list = []
    face_mask = [] # np.array([[[]]])
    # while(cap.isOpened()):
    i = 0
    while(cap.isOpened()):
        # Read frame
        ret, frame = cap.read()
        if ret == True: 
            if i == 0: 
                width,height,z = frame.shape
            
            face_mask_frame = np.zeros((width, height))
            # Estimate movement
            fgMask = backSub.apply(frame) 
            # Image to grascale for facial recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Get guess for face
            faces = face_cascade.detectMultiScale(gray, 1.1)
        
            weight = None
            for (x, y, w, h) in faces:
                #Check if face moved
                weight = fgMask[y:y+h, x:x+w]/255
                weight = sum(sum(weight)) / np.size(weight)
                if weight > 0.01: #If face moved
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    #add face to mask
                    face_mask_frame[y:y+h, x:x+h] = 1
            
            face_mask.append(face_mask_frame)
        
            cv2.imshow('frame', frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    #%% Mask identifying faces is noisy - it guesses parts that aren't faces
    # Erode faces mask in time dimension (based on the assumption that the 
    # face moves slowly frame by frame.

    face_mask_array = np.reshape(face_mask, (len(face_mask), width, height)).astype('uint8')
    struct = np.array([[[True]], [[True]], [[True]]])
    face_mask_erode  = ndimage.binary_erosion(face_mask_array, structure=struct, iterations=4).astype('uint8')
    struct = ndimage.morphology.generate_binary_structure(3,1)
    face_mask_erode = ndimage.binary_dilation(face_mask_erode, structure=struct, iterations=4).astype('uint8')
    struct[(0,2),1,1] = False
    face_mask_erode = ndimage.binary_dilation(face_mask_erode, structure=struct, iterations=20).astype('uint8')
    
    #%% Apply mask, save videio
    cap = cv2.VideoCapture(file)
    
    out = cv2.VideoWriter('prc/'+file,  -1, 20.0, (width,height))
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:       
            mask = face_mask_erode[i,:,:]
        
            frame2 = cv2.bitwise_or(frame, frame, mask = mask)
    
            cv2.imshow('frame', frame2)
            cv2.waitKey(1)
            out.write(frame2)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


