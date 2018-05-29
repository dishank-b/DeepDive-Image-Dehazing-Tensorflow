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
		self.model_path = model_path
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"
		if not os.path.exists(model_path):
			os.makedirs(self.graph_path+"train/")
			os.makedirs(self.graph_path+"val/")
		

	def moduleA(self, x):
		with tf.name_scope("Module_A") as scope:
			with tf.variable_scope("Module_A") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=6, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase, 
								add_summary=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=6, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase, 
								add_summary=True, name="Conv1_2")
				conv1_3 = Conv_2D(x, output_chan=8, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase, 
								add_summary=True, name="Conv1_3")

				conv2_1 = Conv_2D(conv1_1, output_chan=6, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv2_1")
				conv2_2 = Conv_2D(conv1_3, output_chan=12, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv2_2")

				conv3_1 = Conv_2D(conv2_2, output_chan=16, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv3_1")

				concat = tf.concat([conv1_2, conv2_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				outA = tf.nn.relu(ele_sum)

				return outA


	def moduleB(self, x):
		with tf.name_scope("Module_B") as scope:
			with tf.variable_scope("Module_B") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase,
						add_summary=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase,
						add_summary=True, name="Conv1_2")

				conv2_1 = Conv_2D(conv1_2, output_chan=28, kernel=[1,3], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv2_1")
				conv3_1 = Conv_2D(conv2_1, output_chan=32, kernel=[3,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv3_1")

				concat = tf.concat([conv1_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				outB = tf.nn.relu(ele_sum)

				return outB

	def moduleC(self, x):
		with tf.name_scope("Module_C") as scope:
			with tf.variable_scope("Module_C") as var_scope:
				conv1_1 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase,
					add_summary=True, name="Conv1_1")
				conv1_2 = Conv_2D(x, output_chan=24, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,train_phase=self.train_phase,
					add_summary=True, name="Conv1_2")

				conv2_1 = Conv_2D(conv1_2, output_chan=28, kernel=[1,7], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv2_1")
				conv3_1 = Conv_2D(conv2_1, output_chan=32, kernel=[7,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv3_1")

				concat = tf.concat([conv1_1, conv3_1], axis=3, name="concat")		
				
				conv4_1 = Conv_2D(concat, output_chan=16, kernel=[1,1], stride=[1,1], padding="SAME", use_bn=True, activation=None,
						add_summary=True, train_phase=self.train_phase, name="Conv4_1")

				ele_sum = tf.add(x, conv4_1, name="Residual_Sum")

				outC = tf.nn.relu(ele_sum)

				return outC

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.x = tf.placeholder(tf.float32, shape=[None,224,224,3], name="Input")
			self.y = tf.placeholder(tf.float32, shape=[None,224,224,3], name="Output")
			self.train_phase = tf.placeholder(tf.bool, name="is_training")

		with tf.name_scope("Model") as scope:
			conv1  = Conv_2D(self.x, output_chan=16, kernel=[3,3], stride=[1,1], padding="SAME", use_bn=True,
							add_summary=True, train_phase=self.train_phase, name="Conv1")
			modA = self.moduleA(conv1)
			modB = self.moduleB(modA)
			modC = self.moduleC(modB) 
			self.output = Conv_2D(modC, output_chan=3, kernel=[3,3], stride=[1,1], padding="SAME", activation=L_BReLU,use_bn=True, 
								add_summary=True, train_phase=self.train_phase, name="output")
			
			vgg_net1 = Vgg16("./vgg16.npy")
			vgg_net1.build(self.y)
			
			vgg_net2 = Vgg16("./vgg16.npy")
			vgg_net2.build(self.output)

		with tf.name_scope("Loss") as scope:
			
			self.loss = tf.losses.mean_squared_error(self.y, self.output) #\
					  	# + tf.losses.mean_squared_error(vgg_net1.conv3_3/255.0, vgg_net2.conv3_3/255.0)

			self.train_loss_summ = tf.summary.scalar("Loss", self.loss)

		with tf.name_scope("Optimizers") as scope:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  			with tf.control_dependencies(update_ops):
				self.solver = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(self.loss)

		self.merged_summ = tf.summary.merge_all()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.train_writer = tf.summary.FileWriter(self.graph_path+"train/")
		self.train_writer.add_graph(self.sess.graph)
		self.val_writer = tf.summary.FileWriter(self.graph_path+"val/")
		self.val_writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		# tf.train.write_graph(self.sess.graph_def, self.model_path, 'modelGraphDef.pbtxt')
		self.sess.run(tf.global_variables_initializer())

	def train_model(self, train_imgs, val_imgs, learning_rate=1e-5, batch_size=32, epoch_size=50):
		self.debug_info()
		print "Training Images: ", train_imgs.shape[0]
		print "Validation Images: ", val_imgs.shape[0]
		print "Training is about to start with"
		print "Learning_rate: ", learning_rate, "Batch_size", batch_size, "Epochs", epoch_size
		with tf.name_scope("Training") as scope:
			for epoch in range(epoch_size):
				for itr in xrange(0, train_imgs.shape[0]-batch_size, batch_size):
					in_images = train_imgs[itr:itr+batch_size][1]
					out_images = train_imgs[itr:itr+batch_size][0]

					sess_in = [self.solver ,self.loss, self.merged_summ]
					sess_out = self.sess.run(sess_in, {self.x:in_images,self.y:out_images,self.train_phase:True})
					self.train_writer.add_summary(sess_out[2])

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr, "Loss: ", sess_out[1]

				for itr in xrange(0, val_imgs.shape[0]-batch_size, batch_size):
					in_images = val_imgs[itr:itr+batch_size][1]
					out_images = val_imgs[itr:itr+batch_size][0]

					val_loss, summ = self.sess.run([self.loss, self.merged_summ], {self.x: in_images, self.y: out_images,self.train_phase:False})
					self.val_writer.add_summary(summ)

					print "Epoch: ", epoch, "Iteration: ", itr, "Validation Loss: ", val_loss

				if epoch%20==0:
					self.saver.save(self.sess, self.save_path+"DeepDive", global_step=epoch)
					print "Checkpoint saved"

					random_img = train_imgs[np.random.randint(1, train_imgs.shape[0], 4)]

					gen_imgs = self.sess.run([self.output], {self.x: random_img[:,1,:,:,:],self.train_phase:False})

					for i in range(2):
						image_grid_horizontal = 255.0*random_img[i*2][1]
						image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*random_img[i*2][0]))
						image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[0][i*2]))
						for j in range(1):
							image = 255.0*random_img[i*2+1][1]
							image_grid_horizontal = np.hstack((image_grid_horizontal, image))
							image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*random_img[i*2+1][0]))
							image_grid_horizontal = np.hstack((image_grid_horizontal, 255.0*gen_imgs[0][i*2+1]))
						if i==0:
							image_grid_vertical = image_grid_horizontal
						else:
							image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

					cv2.imwrite(self.output_path +str(epoch)+"_train_img.jpg", image_grid_vertical)

	def debug_info(self):
		variables_names = [[v.name, v.get_shape().as_list()] for v in tf.trainable_variables()]
		print "Trainable Variables:"
		tot_params = 0
		for i in variables_names:
			var_params = np.prod(np.array(i[1]))
			tot_params += var_params
			print i[0], i[1], var_params
		print "Total number of Trainable Parameters: ", str(tot_params/1000.0)+"K"