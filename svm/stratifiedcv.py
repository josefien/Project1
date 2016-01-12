import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction')
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/util')
import feature_extraction as feat_ext
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from scikitsvm import Scikit_SVM
from sklearn.metrics import confusion_matrix
import numpy as np
import data_preprocessing
import output_util
import matplotlib.pyplot as plt

dataset_path = 'C:\\Users\\Wim\\Documents\\AIDKE\\Project 1\\Data set\\foodimages\\foodimages'

all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']

""" Class that runs stratified k-fold cross validation on a data set
	whose name is given by parameter string 'dataset'.
"""
class StratifiedCrossValidator(object):
	"""docstring for StratifiedCrossValidator"""
	def __init__(self,n_folds,clf,dataset):
		super(StratifiedCrossValidator, self).__init__()
		self.clf = clf
		self.n_folds = n_folds
		self.dataset = dataset
		print("Setting up validator...")

	# Ask the classifier to provide the data and response values that will be used to train
	# and validate the data. The classifier will have to take care of any data preprocessing
	# steps by itself. I factored this out because SVM and CNN will have a different process
	# of preprocessing data.
	# dataset: string of which dataset will be used for the SVM (determines which feature and 
	# label files will be loaded)
	def _load_data(self,dataset):
		self.X, self.y = self.clf.load_data(dataset)

	# Divide labels/responses into n stratified folds.
	# Stratification means that each folds has approximately the same distribution of labels/responses.
	# StratifiedKFold is a built-in class in Scikit-learn.
	def _stratify(self):
		print("Stratifying samples...")
		self.fold_indices = StratifiedKFold(self.y, self.n_folds)

	# Main loop of the cross-validation algorithm. Data is loaded and stratified; classifier is
	# trained and tested on each k-fold stratified data set.
	# Finally, the average of the attained scores is returned.
	def run(self):
		self._load_data(self.dataset)

		print('Normalizing data...')
		# Scale/normalize data to [0,1] range.
		# The scaler that is "fit" to the training data could be used to scale
		# the test data along the same scale as the training data. It is not being used
		# right now because it would conflict with the chi2 kernel since it produces
		# negative values which the chi2 kernel cannot work with.
		scaler, self.scaled_data = data_preprocessing.getScaler(self.X)
		self._stratify()
		scores = []
		confusion_matrices = []
		i = 1
		for train, test in self.fold_indices:
			print('Start classifier no. {}...'.format(i))
			# Assign training and test data sets based on fold indices
			training_data, training_response, test_data, test_response = self.scaled_data[train], self.y[train], self.scaled_data[test], self.y[test]

			print('Training classifier number {}'.format(i))
			# Train classifier
			self.clf.train(training_data,training_response)

			# Predict labels for computing confusion matrix
			test_prediction = self.clf.predict(test_data)

			# Compute confusion matrix for this fold
			confusion_matrices.append(confusion_matrix(test_response,test_prediction))

			print('Testing classifier number {}'.format(i))
			# Test classifier and store its performance score on the test data set
			scores.append(self.clf.score(test_data,test_response))

			i = i + 1
		# Print the average accuracy (over all labels) for each fold
		print("scores:\n{}".format(scores))

		return self._compileConfusionMatrices(confusion_matrices)

	def _compileConfusionMatrices(self,matrices):
		matrix_sum = matrices.pop()
		for matrix in matrices:
			matrix_sum = np.add(matrix_sum,matrix)
		return matrix_sum

""" Function for testing the plotting of the normalized confusion matrix """
def test_cm_plotting():
	non_normalized_cm = np.loadtxt('confusion_matrix.txt',np.float32)
	cm_normalized = non_normalized_cm.astype('float') / non_normalized_cm.sum(axis=1)[:, np.newaxis]
	output_util.plot_confusion_matrix(cm_normalized,all_labels,title='Normalized Confusion Matrix')

""" Tests the entire cross-validation pipeline, including training, testing and visualization """
def test_entire_pipeline():
	kernel = Scikit_SVM.getLinearKernel()
	clf = Scikit_SVM(kernel)
	validator = StratifiedCrossValidator(10,clf)
	cm = validator.run()
	norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	output_util.plot_confusion_matrix(norm_cm,all_labels,title='Normalized Confusion Matrix')

if __name__ == '__main__':
