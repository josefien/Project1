import sys
sys.path.append('C:/Users/Nadine/git/Project1/util')
from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal

path= 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\features\\feature_methods\\'
features_to_join = [path+'gabor/standard_features.txt',path+'hist/standard_features.txt']
or_file = path + 'hist_gabor/standard_features.txt'

X = np.loadtxt(features_to_join[0],np.float32)
	
for i in range(1,len(features_to_join)):
	print i
	feature_matrix = np.loadtxt(features_to_join[i],np.float32)
	X = np.concatenate((X,feature_matrix),axis=1)

np.savetxt(or_file,X)
