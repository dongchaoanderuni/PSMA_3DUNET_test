# -*- coding: utf-8 -*-

import pickle
import h5py
import numpy as np
import tables

with open('./data/training_ids.pkl', 'rb') as f:
    traingingdata = pickle.load(f)
with open('./data/validation_ids.pkl', 'rb') as f:
    validationdata = pickle.load(f)

#f1 = h5py.File('./data/BAT_data.h5','r')
## List all groups
#print("Keys: %s" % f1.keys())
#a_group_key = list(f1.keys())[0]
#
## Get the data
#data = list(f1[a_group_key])
    
data_file = tables.open_file('./data/BAT_data.h5','r')
data_arr = np.asarray([data_file.root.data[0]])

