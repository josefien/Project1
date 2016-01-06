from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal

all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
labels = ['Boterhammen','Fruit','Aardappelen','Yoghurt','Salade']
labels = all_labels
counters = np.zeros(len(labels),int)

# Counting occurences of the labels as single label

if __name__ == '__main__':
	f = open('classes.txt', 'r')
	f2 = open('frequencies.txt','w')
	for line in f:

	 	classes = line.split('\t')[1].split(',')
	 	classes_v2 = []
	 	for class_ in classes:
	 		class_ = class_.strip()
	 		classes_v2.append(class_)
	 	for i in range(len(labels)):
	 		if labels[i] in classes_v2:
	 			if len(classes_v2) == 1:
					counters[i] = counters[i] + 1
	for i in range(len(labels)):
		f2.write("%s :: %d\n" %(labels[i],counters[i]))