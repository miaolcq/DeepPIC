# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:06:07 2022

@author: tianmiao
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
from multiprocess import Pool
from numba import jit
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys
sys.path.append("D:/Dpic/git/model")
from model import *
from diff_instruments import *
from true import *

path = "D:/Dpic/data2/leaf_seed/data/10.mzXML"
choose_spec0, rt, rt_mean_interval = readms(path)
choose_spec = get_range(choose_spec0, mass_inv = 1, rt_inv = 15, min_intensity=6000)#ninanjie
p = Pool(5)
array = p.map(get_array,choose_spec)
input_data = Input(shape=(256, 256, 1))
model = get_unet(input_data, n_filters=64, dropout=0.5, batchnorm=True, padding='same')
model.compile(optimizer=Adam(lr = 0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.load_weights('D:/Dpic/data/train/best_unet2_zz.h5')
preds = model.predict(scaler(array),batch_size=4,verbose=1)
pred_array_int = pred_array(0, preds, array)
pred_array_mz = pred_array(2, preds, array)
pics = pics(array,pred_array_int,pred_array_mz,choose_spec)



