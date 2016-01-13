import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/util')
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/svm')
from dataloader import DataLoader
from scikitsvm import Scikit_SVM
from stratifiedcv import StratifiedCrossValidator
import numpy as np

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
	def run(dataset_string, clfs, n_folds):
		X,y = DataLoader(dataset_string).load_data()
		return [StratifiedCrossValidator(n_folds,clf,X,y).run() for clf in clfs]

	""" Produce output from experiments: write results to file,
	produce plots, etc.
	"""
	@staticmethod
	def output(conf_mats,dataset_string,clf_names):
		i = 0
		for mat in conf_mats:
			filename = 'outputtest1_{}_{}.txt'.format(dataset_string,clf_names[i])
			np.savetxt(filename,mat)
			i = i + 1

def example():
	# The string indicating the data set that is going to be used for this experiment
	dataset_string = 'standard'

	# The values for the SVM's penalty parameter C (these are just dummy values)
	C_params = [0.3, 0.5, 1.0]

	# Possible values for the chi^2 kernel's gamma parameter (again no idea if these are plausible values)
	chi2_params = [0.3, 0.5, 1.0]
	# Create chi^2 kernels with these values
	chi2kernels = [Scikit_SVM.getChi2Kernel(p) for p in chi2_params]
	# Create strings representing these kernel/parameter combinations
	chi2_clf_names = ['{}_C_{}_chi2_{}'.format(dataset_string,C,p) for C in C_params for p in chi2_params]
	# Create chi2 classifiers
	chi2_clfs = [Scikit_SVM(kernel,C) for C in C_params for kernel in chi2kernels]

	# Linear kernel has no parameters apart from the SVM's own C parameter
	linear_kernel = Scikit_SVM.getLinearKernel()
	# Create strings representing these kernel/parameter combinations
	linear_clf_names = ['{}_C_{}_linear'.format(dataset_string,C) for C in C_params]
	# Create linear classifiers
	lin_clfs = [Scikit_SVM(linear_kernel,C) for C in C_params]

	# Concatenate classifiers into one list
	clfs = chi2_clfs + lin_clfs
	# Concatenate classifier names into one list
	clf_names = chi2_clf_names + linear_clf_names

	n_folds = 10

	Experiment.output(Experiment.run(dataset_string,clfs,n_folds),dataset_string,clf_names)

if __name__ == '__main__':
	example()