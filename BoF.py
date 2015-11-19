import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from image_loader import *

datapath = 'C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages'
# Create image loader
loader = ImageLoader('image_classification.csv',datapath)

class BoF:
    def __init__(self, datapath, loader):
        self.datapath = datapath
        self.loader = loader

        # Create feature detection and keypoint extractor objects (from OpenCV)
        self.fea_det = cv2.FeatureDetector_create("SIFT")
        self.des_ext = cv2.DescriptorExtractor_create("SIFT")


    def createBagOfWords():
        # Create set of all image descriptors from training image set
        des_list = __createKptsDscrpts()

        # Cluster descriptors to create "vocabulary" of visual words
        # over image training set
        k = 3
        self.vocab, variance = __clusterVisualWords(k, des_list)

        # For each image i, create its histogram of features that will represent
        # the image and will be used for classification.
        # This is done by assigning each descriptor in i its nearest
        # visual "word" in the vocabulary calculated in the previous step.
        self.im_features = __createHistOfFeatures(des_list, vocab, k)

    # Create descriptors from images in training set
    def __createKptsDscrpts(self):
        # Start reading in images from training set
        self.loader.startIteration()
        dscrpts = []
        while self.loader.hasNext():
            if i%50 == 0:
                print 'Processing image no. ' + str(i)
            # Read in next image
            [im,classes,image_path] = self.loader.getNextImage()
            # Detect keypoints
            kpts = fea_det.detect(im)
            # Create descriptors from keypoints
            kpts, des = des_ext.compute(im, kpts)
            # Store descriptors
            dscrpts.append((image_path, des))
        self.loader.closeIteration()
        return dscrpts

    def __clusterVisualWords(self, des_list, k):
        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]
        for _, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        # vocab is the vocabulary of visual "words" or descriptors
        vocab, variance = kmeans(descriptors, k, 1)
        return vocab, variance

    def __createHistOfFeatures(self, des_list, vocab, k):
        # Calculate the histogram of features
        feats = np.zeros((len(des_list), k), "float32")
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
                feats[i][w] += 1
        return feats


# Perform Tf-Idf vectorization
# see (https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for full explanation
# Basically term frequency (tf) and inverse document frequency (idf) are weighting factors that scale the
# words in the bag of features vocabulary based on their frequency in the training data set.
def tfIdf():
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')

def standardize():
    # Standardize features: center by removing mean and scaling to unit variance
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)


np.savetxt('test.txt',im_features)