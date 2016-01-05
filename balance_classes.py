from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal

loader = ImageLoader('image_classification.csv','C:\Users\Nadine\Documents\University\Uni 2015\RPMAI1\\foodimages\\foodimages')
all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']

if __name__ == '__main__':

	# Maximum number of instances aloud per class
	max_per_class = 400
	labels = ['Boterhammen','Fruit','Aardappelen','Yoghurt','Salade']
	counters = np.zeros(len(labels),int)
	loader.startIteration()
	for i in range(2000):
	#while loader.hasNext():		
		[img, classes, classpath] = loader.getNextImage() 
		#print(classes)
		for i in range(len(labels)):
			if labels[i] in classes:
				# Possible to seperate single-label items
				#if len(classes) == 1:
					if counters[i] < 400:
						# Use this image
						print labels[i]
						counters[i] = counters[i] + 1
	loader.closeIteration()
	print counters
   