from sklearn import preprocessing
import numpy as np

""" Returns:
		1. a scaler object that is trained on the given data matrix.
		This same scaler can then be used to scale the test data on the
		same scale.
		2. the data matrix, scaled along the feature axis.
"""
def getScaler(data):
	scaler = preprocessing.MinMaxScaler()
	scaled_data = scaler.fit_transform(data)
	return scaler, scaled_data


def minmaxscalertest():
	data = np.random.randn(5,5)
	print('data:\n{}'.format(data))
	print('data[0][2]: {}\n'.format(data[0][2]))
	print('data[1][1]: {}\n'.format(data[1][1]))
	
	scaler = preprocessing.MinMaxScaler()
	norm_data = scaler.fit_transform(data)

	print('norm_data:\n{}'.format(norm_data))

	test_data = np.random.randn(3,5)
	print('test_data\n{}'.format(test_data))

	scaled_test_data = scaler.transform(test_data)
	print('scaled_test_data\n{}'.format(scaled_test_data))

if __name__ == '__main__':
	minmaxscalertest()