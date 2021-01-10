#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:19:04 2021

@author: Thore
"""
del pp
from unfake import pre_process as pp

#%%
path = 'resources/'
#get face locations
pp.process_files_to_get_faces(path)
pp.setup_data(root = path)
