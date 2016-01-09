from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
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
		X = self.kernel(X)
		self.svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='precomputed',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
		self.svm = SVC(kernel='precomputed').fit(X,responses)

	def predict(self,samples):
		return 
		pass

	""" Returns a chi-squared kernel where parameter gamma has been set
		to parameter value.
	"""
	@staticmethod
	def getChi2Kernel(self,gamma_val):
		return functools.partial(chi2_kernel,gamma=gamma_val)

	""" Returns a linear kernel
	"""
	@staticmethod
	def getLinearKernel(self,gamma_val):
		return functools.partial(linear_kernel)

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
	#gammas=[]
	#gammas.append(0.5)
	#kernels=[]
	#for gm in gammas:
	#	kernels.append(getChi2Kernel(gm))

	clf = Scikit_SVM('chi')
	X,y = clf.load_data('standard')
	print(len(X))
	print(len(y))
