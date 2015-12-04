import cv2
import numpy as np

#
class StatModel(object):  
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

# SVM Wrapper class
class SVM(StatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       degree = 1,
                       gamma = 5,
                       coef0 = 1,
                       svm_type = cv2.SVM_C_SVC,
                       C = 2 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        #return self.model.predict_all(samples).ravel()
        return np.float32( [self.model.predict(s) for s in samples])

def fetchLabels(class_path_file):
	f = open(class_path_file,'r')
	labels = []
	for line in f:
		info = line.split('\t')
		label = info[0]
		labels.append(label)
	return np.asarray(labels,dtype='int32')

if __name__ == '__main__':
    labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
    label = labels[0]
    print(label)
    training_features = np.loadtxt('training_feature_'+label+'.txt',np.float32)
    training_labels = fetchLabels('training_label_'+label+'.txt')
    clf = SVM()
    clf.train(training_features, training_labels)
    test_features = np.loadtxt('test_feature_'+label+'.txt',np.float32)
    pred_labels = clf.predict(test_features)
    test_labels = fetchLabels('test_label_'+label+'.txt')
    rFile = open('result.txt','w')
    for i in range(pred_labels.shape[0]):
        rFile.write("%d %d\n" %(test_labels[i],pred_labels[i]))
     