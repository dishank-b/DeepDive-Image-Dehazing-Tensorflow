# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import glob
import sys
import os
from models import *
import yaml


######### Loading Data ###########
blur_images = np.load("./Flower_Images.npy")
blur_images = blur_images[:57]
norm_images = np.load("./Norm_Flower_Images.npy")
print "Data Loaded"
blur_images = 1/127.0*(blur_images-127.0)
norm_images = 1/127.0*(norm_images-127.0)


######## Making Directory #########
log_dir = "./logs/"
model_path = log_dir+sys.argv[1]

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(model_path+"/results")
    os.makedirs(model_path+"/tf_graph")
    os.makedirs(model_path+"/saved_model")


####### Reading Hyperparameters #####$
with open("config.yaml") as file:
	data = yaml.load(file)
	training_params = data['training_params']
	learning_rate = float(training_params['learning_rate'])
	batch_size = int(training_params['batch_size'])
	epoch_size = int(training_params['epochs'])

os.system('cp config.yaml '+model_path+'/config.yaml')

Unet = UNET(model_path)
Unet.build_model()
Unet.train_model(inputs = [norm_images, blur_images],learning_rate, batch_size, epoch_size)

