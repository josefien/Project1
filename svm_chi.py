import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel

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
    
    # Training
    svm = SVC(kernel=chi2_kernel, gamma=.5).fit(training_features, training_labels)
    
    # Predicting 
    test_features = np.loadtxt('test_feature_'+label+'.txt',np.float32)
    pred_labels = svm.predict(test_features)
    test_labels = fetchLabels('test_label_'+label+'.txt')

    rFile = open('result_chi.txt','w')
    for i in range(pred_labels.shape[0]):
        rFile.write("%d %d\n" %(test_labels[i],pred_labels[i]))
     