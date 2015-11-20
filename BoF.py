import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from image_loader import *

class BoF:
    def __init__(self, loader, vocab_size):
        self.loader = loader
        self.vocab_size = vocab_size

        # Create feature detection and keypoint extractor objects (from OpenCV)
        self.fea_det = cv2.FeatureDetector_create("SIFT")
        self.des_ext = cv2.DescriptorExtractor_create("SIFT")


    def createBagOfWords(self):
        # Create set of all image descriptors from training image set
        # (while removing None elements)
        des_list = self._createKptsDscrpts()

        # Cluster descriptors to create "vocabulary" of visual words
        # over image training set
        vocab, variance = self._clusterVisualWords(des_list, self.vocab_size)

        # For each image i, create its histogram of features that will represent
        # the image and will be used for classification.
        # This is done by assigning each descriptor in i its nearest
        # visual "word" in the vocabulary calculated in the previous step.
        self._im_features = self._createHistOfFeatures(des_list, vocab, self.vocab_size)

        self._standardize()

    def getFeatureSet(self):
        return self._im_features

    # Create descriptors from images in training set
    def _createKptsDscrpts(self):
        # Start reading in images from training set
        self.loader.startIteration()
        dscrpts = []
        for i in range(200): # while self.loader.hasNext():
            if i%25 == 0:
                print 'Processing image no. ' + str(i)
            # Read in next image
            [im,classes,image_path] = self.loader.getNextImage()
            # Detect keypoints
            kpts = self.fea_det.detect(im)
            # Create descriptors from keypoints
            kpts, des = self.des_ext.compute(im, kpts)
            if des != None:
                # Store descriptors
                dscrpts.append((image_path, des))
        self.loader.closeIteration()
        return dscrpts

    def _clusterVisualWords(self, des_list, k):
        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]
        i = 0
        for _, descriptor in des_list[1:]:
            #print str(len(descriptor))
            #print 'currently at: ' + str(i)
            #i = i + 1
            if descriptor != None:
                descriptors = np.vstack((descriptors, descriptor))
        # vocab is the vocabulary of visual "words" or descriptors
        vocab, variance = kmeans(descriptors, k, 1)
        return vocab, variance

    def _createHistOfFeatures(self, des_list, vocab, k):
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
    def tfIdf(feats, des_list):
        nbr_occurences = np.sum( (feats > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')
        return idf

    def _standardize(self):
        # Standardize features: center by removing mean and scaling to unit variance
        stdSlr = StandardScaler().fit(self._im_features)
        _im_features = stdSlr.transform(self._im_features)

def _test():
    datapath = 'C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages'
    vocab_size = 50
    # Create image loader
    loader = ImageLoader('image_classification.csv',datapath)
    bof = BoF(loader,vocab_size)
    bof.createBagOfWords()
    feats = bof.getFeatureSet()
    print str(feats.shape)
    np.savetxt('test-20-11.txt',feats)

def main():
    _test()

if __name__ == '__main__':
    main()