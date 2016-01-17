import sys
import cv2
import numpy as np

# Path to directory containing the dataset-directories with feature-files
path = 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\features\\'
# Dataset to be used
dataset = 'dataset1'
# Which size is used, can be ['all','balanced_100','balanced_200']
size = 'balanced_100'
# Which features will be joined, can be ['bof','gabor','hist']
features_to_join = ['bof','gabor']

s1 = ("_").join(list(features_to_join[0]))
f1 = path + dataset + '/' + s1 + '_' + dataset + '_' + size + '_features.txt'
X = np.loadtxt(f1,np.float32)

or_string = s1	
for i in range(1,len(features_to_join)):
	s2 = ("_").join(list(features_to_join[i]))
	or_string = or_string + '_' + s2
	f2 = path + dataset + '/' + s2 + '_' + dataset + '_' + size + '_features.txt'
	feature_matrix = np.loadtxt(f2,np.float32)
	X = np.concatenate((X,feature_matrix),axis=1)

# Result is written to new file with combined prefixes
or_file = path + dataset + '/' + or_string + '_' + dataset + '_' + size + '_features.txt'
np.savetxt(or_file,X)

cf = path + dataset + '/' + s1 + '_' + dataset + '_' + size + '_classes.txt'
ocf = path + dataset + '/' + or_string + '_' + dataset + '_' + size + '_classes.txt'

cf1 = open(cf,'r')
cf2 = open(ocf,'w')

for line in cf1:
	cf2.write(line)