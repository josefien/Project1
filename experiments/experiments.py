import sys
sys.path.append('C:/Users/Wim/Documents/AIDKE/Project 1/New Code/util')
from dataloader import DataLoader

class Experiments:

	""" Load data, parameters for classifiers, etc.
	"""
	@staticmethod
	def setup(dataset_string,clfs,n_folds):
		data_loader = DataLoader(dataset_string)
		X,y = data_loader.load_data()
		return [StratifiedCrossValidator(n_folds,clf,X,y) for clf in clfs]

	""" Train and test classifiers using cross validation """
	@staticmethod
	def run(validators):

	""" Produce output from experiments: write results to file,
	produce plots, etc.
	"""
	@staticmethod
	def output():
		pass

def main():
	datasets = ['set1','set2','set3']
	kernels = [kern1,kern2,kern3]
