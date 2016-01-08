from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
import functools

class Scikit-SVM:

	def __init__(self,kernel):
		self.kernel = kernel

	def train(self,samples,responses):
		X = self.kernel(X, params['gamma'])
		self.svm = SVC(kernel='precomputed').fit(X,responses)

	def predict(self,samples):
		return 
		pass

	@staticmethod
	def chi2(self,samples,gamma):
		return chi2_kernel(samples,gamma)


	@staticmethod
	def getChi2Kernel(self,gamma_val):
		return functools.partial(chi2_kernel,gamma=gamma_val)

if __name__ == '__main__':
	gammas=[]
	gammas.append(0.5)
	kernels=[]
	for gm in gammas:
		kernels.append(getChi2Kernel(gm))
	chi2_kernel = getChi2Kernel()
