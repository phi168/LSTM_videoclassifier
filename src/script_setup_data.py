#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:19:04 2021

@author: Thore
"""
from unfake import pre_process as pp

#%%
path = 'resources/'
#get face locations
#this saves a corresponding .p file next to each video 
pp.process_files_to_get_faces(path)
#processes the .p files to create chunks of image sequences with just the faces
pp.setup_data(root = path)
