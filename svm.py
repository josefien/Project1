import cv2
import numpy as np

# SVM Wrapper class
class SVM(StatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        #return self.model.predict_all(samples).ravel()
        return np.float32( [self.model.predict(s) for s in samples])

def fetchTrainLabels(class_path_file):
	f = open(class_path_file,r)
	labels = []
	for line in f:
		info = line.split('\t')
		label = info[0]
		labels.append(label)
	return np.asarray(labels)

if __name__ == '__main__':
    training_features = np.loadtxt('training_feature_Boterhammen.txt')
    training_labels = fetchTrainLabels('training_label_Boterhammen.txt')
    print(training_labels)
    clf = SVM()
    clf.train(training_features, training_labels)
    test_features = np.loadtxt('test_feature_Boterhammen.txt')
    pred_labels = clf.predict(test_features)
    print(pred_labels)