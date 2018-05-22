# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *
from vgg16 import *

class DeepDive(object):
	"""DeepDive Model"""
	def __init__(self, model_path):
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"

	def moduleA(self, x):
		with tf.name_scope("Module_A") as scope:
			with tf.variable_scope("Module_A") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=6, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=6, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_2")
				conv1_3 = Conv_2D(x, output_chan=8, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_3")

				conv2_1 = Conv_2D(conv1_1, output_chan=6, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, name="Conv2_1")
				conv2_2 = Conv_2D(conv1_3, output_chan=12, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, name="Conv2_2")

				conv3_1 = Conv_2D(conv2_2, output_chan=16, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, name="Conv3_1")

				concat = tf.concat([conv1_2, conv2_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				return ele_sum


	def moduleB(self, x):
		with tf.name_scope("Module_B") as scope:
			with tf.variable_scope("Module_B") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_2")

				conv2_1 = Conv_2D(conv1_2, output_chan=28, kernel=[1,3], stride=[1,1], padding="SAME", use_bn=True, name="Conv2_1")
				conv3_1 = Conv_2D(conv2_1, output_chan=32, kernel=[3,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv3_1")

				concat = tf.concat([conv1_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				return ele_sum

	def moduleC(self, x):
		with tf.name_scope("Module_C") as scope:
			with tf.variable_scope("Module_C") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=32, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=32, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv1_2")

				conv2_1 = Conv_2D(conv1_2, output_chan=32, kernel=[1,7], stride=[1,1], padding="SAME", use_bn=True, name="Conv2_1")
				conv3_1 = Conv_2D(conv2_1, output_chan=32, kernel=[7,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv3_1")

				concat = tf.concat([conv1_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				return ele_sum


	def build_model(self, batch_size=4):
		with tf.name_scope("Inputs") as scope:
			self.x = tf.placeholder(tf.float32, shape=[None,224,224,3], name="Input")
			self.y = tf.placeholder(tf.float32, shape=[None,224,224,3], name="Output")
			self.train_phase = tf.placeholder(tf.bool, name="Train/Test_Phase")

		with tf.name_scope("Model") as scope:
			conv1  = Conv_2D(self.x, output_chan=16, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, name="Conv1")
			modA = self.moduleA(conv1)
			modB = self.moduleB(modA)
			modC = self.moduleC(modB) 
			self.output = Conv_2D(modC, output_chan=3, kernel=[3,3], stride=[1,1], padding="SAME", activation=BReLU,use_bn=True, name="output")

			vgg_net1 = Vgg16("./vgg16.npy")
			vgg_net1.build(self.y)
			
			vgg_net2 = Vgg16("./vgg16.npy")
			vgg_net2.build(self.output)

		with tf.name_scope("Loss") as scope:
			
			self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.output)) \
						+ tf.reduce_mean(tf.losses.mean_squared_error(vgg_net1.pool4, vgg_net2.pool4)) \
						+ tf.reduce_mean(tf.losses.mean_squared_error(vgg_net1.pool5, vgg_net2.pool5)) \
						+ tf.reduce_mean(tf.losses.mean_squared_error(vgg_net1.relu7, vgg_net2.relu7))

		with tf.name_scope("Optimizers") as scope:
			self.solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.1).minimize(self.loss)

		self.sess = tf.Session()
		self.writer = tf.summary.FileWriter(self.graph_path)
		self.writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())

	def train_model(self, train_imgs, val_imgs, learning_rate=1e-5, batch_size=32, epoch_size=50):

		with tf.name_scope("Training") as scope:
			for epoch in range(epoch_size):
				for itr in xrange(0, train_imgs.shape[0]-batch_size, batch_size):
					in_images = train_imgs[itr:itr+batch_size][1]
					out_images = train_imgs[itr:itr+batch_size][0]

					sess_in = [self.solver ,self.loss]
					sess_out = self.sess.run(sess_in, {self.x:in_images,self.y:out_images,self.train_phase:True})

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr
						print "Loss: ", sess_out[1]

				if epoch%10==0:
					self.saver.save(self.sess, self.save_path)
					print "Checkpoint saved"

					val_loss = self.sess.run([self.loss, self.output], {self.x: val_imgs[:,1,:,:,:], self.y:val_imgs[:,0,:,:,:],
							   self.train_phase:False})

					print "Epoch: ", epoch, "Iteration: ", itr
					print "Loss: ", val_loss

					random_img = train_imgs[np.random.randint(1, train_imgs.shape[0], 4)]
					gen_imgs = self.sess.run([self.output], {self.x: random_img[1],self.train_phase:False})

					for i in range(2):
						image_grid_horizontal = 255.0*random_img[i*2][0]
						image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[i*2]))
						for j in range(1):
							image = 255.0*random_img[i*2+1][0]
							image_grid_horizontal = np.hstack((image_grid_horizontal, image))
							image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[i*2+1]))
						if i==0:
							image_grid_vertical = image_grid_horizontal
						else:
							image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

					cv2.imwrite(self.output_path +"/train_img_"+str(epoch)+".jpg", image_grid_vertical)

					random_img = val_imgs[np.random.randint(1, val_imgs.shape[0], 4)]
					gen_imgs = self.sess.run([self.output], {self.x: random_img[1],self.train_phase:False})

					for i in range(2):
						image_grid_horizontal = 255.0*random_img[i*2][0]
						image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[i*2]))
						for j in range(1):
							image = 255.0*random_img[i*2+1][0]
							image_grid_horizontal = np.hstack((image_grid_horizontal, image))
							image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[i*2+1]))
						if i==0:
							image_grid_vertical = image_grid_horizontal
						else:
							image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

					cv2.imwrite(self.output_path +"/val_img_"+str(epoch)+".jpg", image_grid_vertical)
