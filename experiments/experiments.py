import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/util')
from dataloader import DataLoader

class Experiment:

	""" Load data set, set up and run cross validators.
		Parameter description:
		dataset_string: the string indicating the file name of the feature/data set
		clf_names: a list of strings indicating the name of the classifiers that are
		being run. Can be used for naming files when outputting the confusion matrices. Need
		to be in the same order as the list of classifiers clfs that is passed.
		clfs: a list of SVM's for which kernels and penalty parameter C have already been set.
		n_folds: the number of folds to use for cross validation.
	"""
	@staticmethod
	def run(dataset_string, clf_names, clfs, n_folds):
			X,y = DataLoader(dataset_string).load_data()
			conf_matrices = [StratifiedCrossValidator(n_folds,clf,X,y).run() for clf in clfs]
			output(conf_matrices)

	""" Produce output from experiments: write results to file,
	produce plots, etc.
	"""
	@staticmethod
	def output(conf_mats):
		pass


def main():
	dataset = 'standard'
	kernels = [kern1,kern2,kern3]
	Experiments.run(Experiments.setup())