import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction')
import feature_extraction as feat_ext
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataset_path = 'C:\\Users\\Wim\\Documents\\AIDKE\\Project 1\\Data set\\foodimages\\foodimages'

class StratifiedCrossValidator(object):
	"""docstring for StratifiedCrossValidator"""
	def __init__(self,n_folds,pipeline):
		super(StratifiedCrossValidator, self).__init__()
		self.clf = pipeline
		self.n_folds = n_folds

	@staticmethod
	def get_full_svm_pipeline():
		return Pipeline([StandardScaler(), SVM()])
	
	def _load_data():
		self.X = clf.load_data()
		self.y = clf.load_response()

	# Divide labels/responses into n stratified folds
	# Stratification means that each folds has approximately the same distribution of labels/responses
	def _stratify(self, n_folds):
		self.fold_indices = StratifiedKFold(self.y, n_folds)
		pass

	def run(self, classifier):
		scores = []
		for train, test in self.fold_indices:
			training_data = self.X[train]
			training_response = self.y[train]
			test_data = self.X[test]
			test_response = self.y[test]

			kth_classifier = self.clf.setparams(PLACEHOLDER)
			kth_classifier.fit(training_data,training_response)

			scores.append(kth_classifier.score(test_data,test_response))
