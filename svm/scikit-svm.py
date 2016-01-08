from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
import functools

class Scikit-SVM:

	def __init__(self,kernel):
		self.kernel = kernel

	def train(self,samples,responses):
		X = self.kernel(X)
		self.svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='precomputed',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
		self.svm = SVC(kernel='precomputed').fit(X,responses)

	def predict(self,samples):
		return 
		pass

	""" Returns a chi-squared kernel where parameter gamma has been set
		to parameter value.
	"""
	@staticmethod
	def getChi2Kernel(self,gamma_val):
		return functools.partial(chi2_kernel,gamma=gamma_val)

	""" Returns a linear kernel
	"""
	@staticmethod
	def getLinearKernel(self,gamma_val):
		return functools.partial(linear_kernel)

if __name__ == '__main__':
	gammas=[]
	gammas.append(0.5)
	kernels=[]
	for gm in gammas:
		kernels.append(getChi2Kernel(gm))
