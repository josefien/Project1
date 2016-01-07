import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction')
import feature_extraction as feat_ext
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataset_path = 'C:\\Users\\Wim\\Documents\\AIDKE\\Project 1\\Data set\\foodimages\\foodimages'

# Class that runs stratified k-fold cross validation on a data set.
class StratifiedCrossValidator(object):
	"""docstring for StratifiedCrossValidator"""
	def __init__(self,n_folds,clf):
		super(StratifiedCrossValidator, self).__init__()
		self.clf = clf
		self.n_folds = n_folds

	# Ask the classifier to provide the data and response values that will be used to train
	# and validate the data. The classifier will have to take care of any data preprocessing
	# steps by itself. I factored this out because SVM and CNN will have a different process
	# of preprocessing data.
	def _load_data():
		self.X = clf.load_samples()
		self.y = clf.load_responses()

	# Divide labels/responses into n stratified folds.
	# Stratification means that each folds has approximately the same distribution of labels/responses.
	# StratifiedKFold is a built-in class in Scikit-learn.
	def _stratify(self, n_folds):
		self.fold_indices = StratifiedKFold(self.y, n_folds)

	# Main loop of the cross-validation algorithm. Data is loaded and stratified; classifier is
	# trained and tested on each k-fold stratified data set.
	# Finally, the average of the attained scores is returned.
	def run(self, classifier):
		self._load_data()
		self._stratify(self.n_folds)
		scores = []
		for train, test in self.fold_indices:
			# Assign training and test data sets based on fold indices
			training_data, training_response, test_data, test_response = self.X[train], self.y[train], self.X[test], self.y[test]

			# Create classifier
			kth_classifier = self.clf.setparams(PLACEHOLDER)

			# Train classifier
			kth_classifier.fit(training_data,training_response)

			# Test classifier and store its performance score on the test data set
			scores.append(kth_classifier.score(test_data,test_response))

		return np.average(scores)