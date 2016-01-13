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
	def run(dataset_string, labels, clfs, n_folds):
		X,y = DataLoader(dataset_string,labels).load_data()
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

def main():
	labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
	dataset_string = 'standard'
	kernels = [Scikit_SVM.getLinearKernel()]
	pen_params = [0.5]
	clfs = []
	clf_names = []
	n_folds = 10
	for ker in kernels:
		for p in pen_params:
			clfs.append(Scikit_SVM(ker,p))
			clf_names.append('linear_{}'.format(str(p)))
	print('len(clfs): {}'.format(len(clfs)))
	Experiment.output(Experiment.run(dataset_string,labels,clfs,n_folds),dataset_string,clf_names)

if __name__ == '__main__':
	main()