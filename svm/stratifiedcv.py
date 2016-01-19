import sys
sys.path.append('C:/Users/Nadine/git/Project1/feature_extraction')
sys.path.append('C:/Users/Nadine/git/Project1/util')
import feature_extraction as feat_ext
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from scikitsvm import Scikit_SVM
from sklearn.metrics import confusion_matrix
import numpy as np
import data_preprocessing
import output_util
import matplotlib.pyplot as plt
import numpy as np

dataset_path = 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\foodimages\\foodimages'
all_labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
labels_by_frequency = ['Sandwich','Fruit','Potatoes','Cookies','Yogurt','Pasta','Salad','Rice','Crisps','Candy Bar','Cornflakes','French Fries','Soup','Muffin','Pastry','Candy','Hamburger','Pizza','Ice Cream']
save_matrix_flag = True
save_report_flag = True

cntr = 1

""" Class that runs stratified k-fold cross validation on a data set
	whose name is given by parameter string 'dataset'.
"""
class StratifiedCrossValidator(object):
	"""docstring for StratifiedCrossValidator"""
	def __init__(self,n_folds,clf,X,y):
		super(StratifiedCrossValidator, self).__init__()
		self.clf = clf
		self.n_folds = n_folds
		self.X = X
		self.y = y

	# Divide labels/responses into n stratified folds.
	# Stratification means that each folds has approximately the same distribution of labels/responses.
	# StratifiedKFold is a built-in class in Scikit-learn.
	def _stratify(self):
		self.fold_indices = StratifiedKFold(self.y, self.n_folds)

	# Main loop of the cross-validation algorithm. Data is loaded and stratified; classifier is
	# trained and tested on each k-fold stratified data set.
	# Finally, the average of the attained scores is returned.
	def run(self):
		global cntr
		print('Validating classifier no. {}'.format(cntr))
		cntr = cntr + 1

		# Scale/normalize data to [0,1] range.
		# The scaler that is "fit" to the training data could be used to scale
		# the test data along the same scale as the training data. It is not being used
		# right now because it would conflict with the chi2 kernel since it produces
		# negative values which the chi2 kernel cannot work with.
		scaler, self.scaled_data = data_preprocessing.getScaler(self.X)
		self._stratify()
		scores = []
		i = 1

		# Stores all actual test labels over all validation folds
		responses = np.empty((0,1))
		# Stores all predicted labels over all validation folds
		predictions = np.empty((0,1))

		for train, test in self.fold_indices:
			# Assign training and test data sets based on fold indices
			training_data, training_response, test_data, test_response = self.scaled_data[train], self.y[train], self.scaled_data[test], self.y[test]

			# Train classifier
			self.clf.train(training_data,training_response)

			# Predict labels for computing confusion matrix
			test_prediction = self.clf.predict(test_data)

			responses = np.vstack((responses,np.reshape(test_response,(-1,1))))
			predictions = np.vstack((predictions,np.reshape(test_prediction,(-1,1))))

			# Test classifier and store its performance score on the test data set
			scores.append(self.clf.score(test_data,test_response))

			i = i + 1

		# Print the average accuracy (over all labels) for each fold
		print("scores:\n{}".format(scores))
		average = sum(scores)/len(scores)
		print("average score: %f" %(average))

		responses = np.asarray(responses)
		predictions = np.asarray(predictions)

		ret = np.hstack((responses,predictions))

		print('ret.shape: {}'.format(ret.shape))

		# Return a 2-d array of shape [test_responses,test_predictions]
		return ret


""" Function for testing the plotting of the normalized confusion matrix """
def test_cm_plotting():
	balanced = True

	non_normalized_cm = np.loadtxt('../experiments/dataset1_all_C_0.2_chi2_0.1_cfsn_matrix.txt',np.float32)
	num_labels = non_normalized_cm.shape[0]
	labels = all_labels
	if balanced:
		labels = labels_by_frequency[:num_labels]
	labels = labels_by_frequency[:num_labels]
	cm_normalized = non_normalized_cm.astype('float') / non_normalized_cm.sum(axis=1)[:, np.newaxis]
	output_util.plot_confusion_matrix(cm_normalized,labels,title='Normalized Confusion Matrix')

""" Tests the entire cross-validation pipeline, including training, testing and visualization """
def test_entire_pipeline():
	kernel = Scikit_SVM.getLinearKernel()
	clf = Scikit_SVM(kernel)
	validator = StratifiedCrossValidator(10,clf)
	cm = validator.run()
	norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	output_util.plot_confusion_matrix(norm_cm,labels_by_frequency,title='Normalized Confusion Matrix')

if __name__ == '__main__':
	test_cm_plotting()