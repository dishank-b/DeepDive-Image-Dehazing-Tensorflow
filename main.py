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
# hazed_img = np.load("./Flower_Images.npy")
# dehazed_img = np.load("./Norm_Flower_Images.npy")
# print "Data Loaded"
# hazed_img = 1/255.0*(blur_images-255.0)
# dehazed_img = 1/255.0*(norm_images-255.0)


######## Making Directory #########
log_dir = "./logs/"
model_path = log_dir+sys.argv[1]
print model_path
if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(model_path+"/results")
    os.makedirs(model_path+"/tf_graph")
    os.makedirs(model_path+"/saved_model")


####### Reading Hyperparameters #####
with open("config.yaml") as file:
	data = yaml.load(file)
	training_params = data['training_params']
	learning_rate = float(training_params['learning_rate'])
	batch_size = int(training_params['batch_size'])
	epoch_size = int(training_params['epochs'])
	model_params= data['model_params']
	descrip = model_params['descrip']
	if len(descrip)==0:
		raise ValueError, "Please give a proper description of the model you are training."

os.system('cp config.yaml '+model_path+'/config.yaml')

DD = DeepDive(model_path)
DD.build_model()
# DD.train_model(inputs = [norm_images, blur_images],learning_rate, batch_size, epoch_size)

