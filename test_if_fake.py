#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:22:12 2020

@author: Thore
"""

from keras.models import load_model
import cv2
import numpy as np
import os
import sys

def test_if_fake(video_file):
    cap = cv2.VideoCapture(video_file)
    filename,_ = video_file.split('.')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      #Load Neural Net
    NN = load_model(os.path.join('model','model.keras'))
    print('Loaded Network')
    #Get required input dimensions
    _,seq_len,width_input,height_input,_ = NN.layers[0].input_shape
    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    out_cap = cv2.VideoWriter(filename+'_processed.avi', fourcc , fps, (width_input, height_input))

    i = 0; 
    sequence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        i += 1
        
        #add frames to sequence until we get right number of frames
        frame = cv2.resize(frame, (width_input, height_input))
        
        sequence.append(frame)
        if i < seq_len: # don't have enough frames, append more
            continue
        else:
            i = 0 # Reset counter
            
        # Get prediction
        y = NN.predict(np.array([sequence]))
        label = 'real' if y[0,0]>y[0,1] else 'fake'   
        
        # Write frames
        for im in sequence:
            cv2.putText(im, label, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            out_cap.write(im)
            cv2.imshow('frame', im)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
        sequence = [] #reset sequence
        
    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()
    print('Finished writing to file')
    
def main():
    file = sys.argv[1]  
    test_if_fake(file)
    
if __name__ == '__main__':
    main()