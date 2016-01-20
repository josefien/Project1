import sys
sys.path.append('../util')
sys.path.append('../feature_extraction')
from image_loader import *
import cv2
import numpy as np
import random
import decimal
import sys
import feature_extraction as fe
from sklearn.externals import joblib
import svm
import grabcut

"""
Given as input in the command line the path to the image which needs to be classified, thus for example:
python classify_new_image.py C:/Users/Nadine/git/Project1/example_images/cookies.jpg
The by the classifier predicted label will be printed.

The SVM-model used is trained on a balanced training-set with 5 labels (Sandwich,Potatoes,Fruit,Cookies,Yoghurt)
using the Gabor and Histogram features and Grabcut as pre-processing step.

"""

def extract_features(classpath):
	print "Reading image..."
	img = cv2.imread(classpath)
	print "Applying grabcut..."
	newpath = classpath.split('.', 1)[0]+'grabcut.jpg'
	cv2.imwrite(newpath,grabcut.modifyImage(img))
	print "Extracting features..."
	gb_vector = []
	gb_list = fe.gabor(newpath)
	gb_vector = np.asarray(gb_list)
	h_vector = []
   	h_vector = fe.histogram(newpath)
	feature_vector = np.concatenate((gb_vector,h_vector))
	return feature_vector

if __name__ == '__main__':
	classpath = sys.argv[1]
	fv = extract_features(classpath)
	nfv = np.reshape(fv,(1,354))
	clf = joblib.load('svm_model.pkl') 
	print "Classifying..."
	label = np.float32(clf.predict(nfv))
	print "Label predicted is:"
	if label == 0:
		print 'Sandwich'
	if label == 1:
		print 'Potatoes'
	if label == 5:
		print 'Fruit'
	if label == 9:
		print 'Cookies'
	if label == 18:
		print 'Yoghurt'