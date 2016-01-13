import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/util')
from dataloader import DataLoader

class Experiment:

	def __init__(self, dataset_string, clf_names, clfs, n_folds):
		self.filename=filename
		self.X, self.y = DataLoader(dataset_string).load_data()

	""" Load data, parameters for classifiers, etc.
	"""
	def setup(dataset_string,clfs,n_folds):
		X,y = data_loader.load_data()
		return [StratifiedCrossValidator(n_folds,clf,X,y) for clf in clfs]

	""" Train and test classifiers using cross validation """
	@staticmethod
	def run(validators):
		return [val.run() for val in validators]

	""" Produce output from experiments: write results to file,
	produce plots, etc.
	"""
	@staticmethod
	def output(conf_mats):
		for mat in conf_mats:


def main():
	datasets = ['set1','set2','set3']
	kernels = [kern1,kern2,kern3]
	Experiments.run(Experiments.setup())