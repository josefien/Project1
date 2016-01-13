from itertools import izip
import re
import numpy as np

class DataLoader:

	def __init__(self,dataset_string):
		self.dataset_string = dataset_string

	""" Reads files outputted by feature_extraction.py
	features.txt contains per line the feature array of the image
	classes.txt contains the image path and the labels (as strings)
	Each line corresponds to the same image
	dataset: string of which dataset will be used for the SVM (determines 
	which feature and label files will be loaded)
	"""
	def load_data(self):
		print("loading data...")
		inputFeatures = '../feature_extraction/' + self.dataset_string + '_features.txt'
		inputLabels = '../feature_extraction/' + self.dataset_string + '_classes.txt'
		print('type(inputFeatures): {}'.format(type(inputFeatures)))
		X,y = self._adjustFeatures(inputFeatures,inputLabels)
		return (X,y)

	""" Reads the feature- and label-file and duplicates all the
	images with multiple labels. For example
	[f e a t u r e] - label1, label2, label3
	becomes
	[f e a t u r e] - label1
	[f e a t u r e] - label2
	[f e a t u r e] - label3
	Returns both extended feature matrix and label vector
	"""
	def _adjustFeatures(self,feature_file,label_file):
		all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
		ff = open(feature_file,'r')
		lf = open(label_file,'r')
		newFF = open('features_extended.txt','w')
		newLF = open('labels_extended.txt','w')
		labels = []
		for featL, labelL in izip(ff,lf):
			info = labelL.split('\t')
			path = info[0]
			label_string = info[1]
			labels = re.sub("\n","",label_string).split(',')
			for i in range(len(all_labels)):
				# Turns labels into numerical values 
				if all_labels[i] in labels:
					newFF.write(featL)
					newLF.write("%d\t%s\n" %(i,path))
		newFF.close()
		newLF.close()
		features = np.loadtxt('features_extended.txt',np.float32)
		labels = self._fetchLabels('labels_extended.txt')
		return (features,labels)
    
	def _fetchLabels(self,class_path_file):
		f = open(class_path_file,'r')
		labels = []
		for line in f:
			info = line.split('\t')
			label = info[0]
			labels.append(label)
		return np.asarray(labels,dtype='int32')