import numpy as np
from os import *
from glob import *
import cv2

def resize_img(x_img_list, y_img_list,name):
	clear_path = "/media/shareit/haze_dataset/clear/"
	haze_path = "/media/shareit/haze_dataset/haze/OTS/"
	
	final_size = 224
	
	x_image_list = []
	y_image_list = []
	
	for image in x_img_list:
		# print image		
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
		# print image
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
	
	npy = []
	for i in range(len(x_image_list)):
		pair = [x_image_list[i], y_image_list[i]]
		npy.append(pair)

	npy = np.array(npy)
	np.random.shuffle(npy)

	print npy.shape
	np.save("/media/shareit/haze_dataset/npy_files/"+name, npy)


def create_npy():
	clear_path = "/media/shareit/haze_dataset/clear/"
	haze_path = "/media/shareit/haze_dataset/haze/OTS/"

	clear_images = listdir(clear_path)
	haze_images = listdir(haze_path)

	valid_imgs1 = [i[:4] for i in haze_images if "1_0.2" in i]
	valid_imgs2 = [i[:4] for i in haze_images if "0.8_0.2" in i]
	
	valid_imgs = list(set(valid_imgs1)&set(valid_imgs2))

	np.random.seed(2000)
	clear_images = np.random.choice(valid_imgs, 3500, replace=False)
	
	train_x = [i+'.jpg' for i in clear_images[:2000]]
	test_x = [i+'.jpg' for i in clear_images[2000:3000]]
	val_x = [i+'.jpg' for i in clear_images[3000:]]

	train_y = [i[:4]+"_0.8_0.2.jpg" for i in train_x] + [i[:4]+"_1_0.2.jpg" for i in train_x]
	test_y = [i[:4]+"_0.8_0.2.jpg" for i in test_x] 
	val_y = [i[:4]+"_0.8_0.2.jpg" for i in val_x]

	train_a = np.array(train_y)
	train_b = np.hstack((train_x, train_x))

	val_a = np.array(val_y)
	val_b = np.array(val_x)

	test_a = np.array(test_y)
	test_b = np.array(test_x)

	resize_img(train_b,train_a, "Train.npy")	
	resize_img(val_b, val_a, "Val.npy")
	resize_img(test_b, test_a, "Test.npy")
	
def test_npy():
	images = np.load("/media/shareit/haze_dataset/npy_files/Val.npy")
	i=1
	j=1
	print len(images)
	for image in images:
		cv2.imshow("Window1", image[0])
		cv2.imshow("Window2", image[1])
		k=cv2.waitKey(0)
		j+=1




# create_npy() 
test_npy()