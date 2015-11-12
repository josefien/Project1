import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from image_loader import *

datapath = 'C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages'
# Create image loader
loader = ImageLoader('image_classification.csv',datapath)

# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

loader.startIteration()
for i in range(10):
    if i%50 == 0:
        print 'Processing image no. ' + str(i)
    [im,classes,image_path] = loader.getNextImage()
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))
loader.closeIteration()

print ', '.join(str(p) for p in des_list)
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for _, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
k = 3
# vocab is the vocabulary of visual "words" or descriptors
vocab, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(des_list), k), "float32")
for i in xrange(len(des_list)):
    # Each descriptor in this entry of the descriptor list is assigned
    # its nearest visual "word".
    # words is a length M array, where M is the number of descriptors for
    # the given image. Each entry in words stores an index to the nearest
    # visual word in the vocabulary.
    words, distance = vq(des_list[i][1],vocab)
    # for each vocabulary index in words, increment the count for that word
    # in the histogram
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Standardize features: center by removing mean and scaling to unit variance
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

np.savetxt('test.txt',im_features)