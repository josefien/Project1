from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel

import functools
import numpy as np
import re
from itertools import izip

class Scikit_SVM:

	def __init__(self,kernel):
		self.kernel = kernel

	# Reads files outputted by feature_extraction.py
	# features.txt contains per line the feature array of the image
	# classes.txt contains the image path and the labels (as strings)
	# Each line corresponds to the same image
	# dataset: string of which dataset will be used for the SVM (determines 
	# which feature and label files will be loaded)
	def load_data(self,dataset):
		inputFeatures = '../features/' + dataset + '_features.txt'
		inputLabels = '../features/' + dataset + '_classes.txt'
		X,y = adjustFeatures(inputFeatures,inputLabels)
		return (X,y)

	def train(self,samples,responses):
		self.svm = SVC(kernel=self.kernel).fit(samples,responses)
		# Make sure we get the one-vs-rest style output that we want
		self.svm.decision_function_shape = "ovr"

	def predict(self,samples):
		return np.float32( [self.svm.predict(s) for s in samples])

	"""Return the mean accuracy of the trained classifier over all labels in the test set.
		This is the only metric we have now, because it's built in, but we might have to look
		into others, such as precision and recall, but these are quite tricky to implement
		for the multi-class case. This paper has some formulas for multi-class performance
		measurement (page 4): http://atour.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
	"""
	def score(self,test_samples,test_responses):
		return self.svm.score(test_samples,test_responses)
	""" Returns a chi-squared kernel where parameter gamma has been set
		to argument value.
	"""
	@staticmethod
	def getChi2Kernel(self,gamma_val):
		return functools.partial(chi2_kernel,gamma=gamma_val)

	"""Returns a linear kernel"""
	@staticmethod
	def getLinearKernel(self):
		return linear_kernel

	"""Returns a polynomial kernel with parameters set to argument values"""
	@staticmethod
	def getPolyKernel(self,coef0,degree):
		return functools.partial(polynomial_kernel,coef0=coef0,degree=degree)

# Reads the feature- and label-file and duplicates all the
# images with multiple labels. For example
# [f e a t u r e] - label1, label2, label3
# becomes
# [f e a t u r e] - label1
# [f e a t u r e] - label2
# [f e a t u r e] - label3
# Returns both extended feature matrix and label vector
def adjustFeatures(feature_file,label_file):
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
	labels = fetchLabels('labels_extended.txt')
	return (features,labels)
    
def fetchLabels(class_path_file):
	f = open(class_path_file,'r')
	labels = []
	for line in f:
		info = line.split('\t')
		label = info[0]
		labels.append(label)
	return np.asarray(labels,dtype='int32')

if __name__ == '__main__':
	# Set up kernels
	gammas=[]
	gammas.append(0.5)
	kernels=[]
	for gm in gammas:
		kernels.append(getChi2Kernel(gm))

	# Load data
	X = np.loadtxt('features.txt',np.float32)


	for ker in kernels:
		svm = Scikit-SVM(ker)
		svm.train(data,responses)
