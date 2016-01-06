from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal
from itertools import izip

"""
Creates dataset selecting only the labels given in array "labels"
with a maximum number of items per labels given by "max_per_class"
"""

loader = ImageLoader('image_classification.csv','C:\Users\Nadine\Documents\University\Uni 2015\RPMAI1\\foodimages\\foodimages')
all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
labels = ['Boterhammen','Fruit','Aardappelen']
counters = np.zeros(len(labels),int)

if __name__ == '__main__':
	f = open('features.txt','r')
	f2 = open('classes.txt','r')
	r1 = open('features_balanced.txt','w')
	r2 = open('classes_balanced.txt','w')

	# Maximum number of instances aloud per class
	max_per_class = 10000
	for features, info in izip(f, f2):
		classes = info.split('\t')[1].split(',')
	 	classes_v2 = []
	 	for class_ in classes:
	 		class_ = class_.strip()
	 		classes_v2.append(class_)
		for i in range(len(labels)):
			if labels[i] in classes_v2:
				if counters[i] < max_per_class:
					counters[i] = counters[i] + 1
					r1.write(features)
					r2.write(info)

	loader.closeIteration()
	print counters