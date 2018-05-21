import numpy as np
from os import *
from glob import *
import cv2

def resize_img(x_img_list, y_img_list):
	clear_path = "/media/shareit/haze_dataset/clear/"
	haze_path = "/media/shareit/haze_dataset/haze/OTS/"
	
	final_size = 224
	
	x_image_list = []
	y_image_list = []
	
	for image in x_img_list:
		print image		
		img = cv2.imread(clear_path+image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		x_image_list.append(final_image)
	print "X done...................."

	for image in y_img_list:
		print image
		img = cv2.imread(haze_path+image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		y_image_list.append(final_image)
	
	print x_image_list.shape, y_image_list.shape


def create_npy():
	clear_path = "/media/shareit/haze_dataset/clear/"
	haze_path = "/media/shareit/haze_dataset/haze/OTS/"

	clear_images = listdir(clear_path)
	haze_images = listdir(haze_path)

	valid_imgs1 = [i for i in haze_images if "1_0.2" in i]
	valid_imgs2 = [i for i in haze_images if "0.8_0.2" in i]
	
	print "0001_0.8_0.2.jpg" in valid_imgs2
	# clear_images = np.random.choice(clear_images, 3500, replace=False)
	
	# train_x = clear_images[:2000]
	# test_x = clear_images[2000:3000]
	# val_x = clear_images[3000:]

	# train_y = [i[:4]+"_0.8_0.2.jpg" for i in train_x] + [i[:4]+"_1_0.2.jpg" for i in train_x]
	# test_y = [i[:4]+"_0.8_0.2.jpg" for i in test_x] + [i[:4]+"_1_0.2.jpg" for i in test_x]
	# val_y = [i[:4]+"_0.8_0.2.jpg" for i in val_x] + [i[:4]+"_1_0.2.jpg" for i in val_x]

	# a = np.array(train_y)
	# b = np.hstack((train_x, train_x))
	# print a.shape, b.shape

	# train_npy = resize_img(b,a)
	

create_npy() 
