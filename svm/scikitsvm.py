from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel

import functools
import numpy as np
import re
from itertools import izip

class Scikit_SVM:

	""" Set up an SVM with given kernel and penalty parameter C """
	def __init__(self,kernel,C=1.0):
		self.svm = SVC(kernel=kernel,C=C)

	""" Train the SVM using given samples and responses """
	def train(self,samples,responses):
		self.svm = self.svm.fit(samples,responses)
		# Make sure we get the one-vs-rest style output that we want
		self.svm.decision_function_shape = "ovr"

	""" Use the classifier to classify given feature vectors """
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
	def getChi2Kernel(gamma_val):
		return functools.partial(chi2_kernel,gamma=gamma_val)

	"""Returns a linear kernel"""
	@staticmethod
	def getLinearKernel():
		return linear_kernel

	"""Returns a polynomial kernel with parameters set to argument values"""
	@staticmethod
	def getPolyKernel(coef0_val,degree_val):
		return functools.partial(polynomial_kernel,coef0=coef0_val,degree=degree_val)

	""" Returns an RBF kernel with parameter gamma set to given value """
	@staticmethod
	def getRBFKernel(gamma_val):
		return functools.partial(rbf_kernel,gamma=gamma_val)


def test():
	# Set up kernels
	gammas=[]
	gammas.append(0.5)
	kernels=[]
	for gm in gammas:
		kernels.append(Scikit_SVM.getLinearKernel())

	# Load data
	X = np.loadtxt('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction/standard_features.txt',np.float32)
	y = np.loadtxt('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction/standard_classes.txt',np.float32)

	for ker in kernels:
		svm = Scikit_SVM(ker,1.0)
		svm.train(X,y)

if __name__ == '__main__':
	test()