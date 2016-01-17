import sys
sys.path.append('C:/Users/Nadine/git/Project1/util')
import cv2
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from image_loader import *

instances = 100

class BoF:
    def __init__(self, loader, vocab_size, surf_threshold):
        self.loader = loader
        self.vocab_size = vocab_size

        # Create feature detection and keypoint extractor objects (from OpenCV)
        self.fea_det = cv2.FeatureDetector_create("SURF")
        self.des_ext = cv2.DescriptorExtractor_create("SURF")

        self.surf = cv2.SURF(surf_threshold)

        # Feature dictionary that will allow the lookup of the computed
        # feature histogram for a given feature
        self.feat_dict = {}

        self._im_features = []

    def createBagOfWords(self):
        # Create set of all image descriptors from training image set
        # (while removing None elements)
        tFile = open('bof_trainset.txt', 'w')
   
        des_list = []

        # Start reading in images from training set
        self.loader.startIteration()
        des_list = []
        empty_feature_list = []
        #for i in range(instances):
        i = 0
        while self.loader.hasNext():
            if i%1000 == 0:
                print 'BoF processing image no. ' + str(i)
            i = i + 1
            # Read in next image
            [im,classes,image_path] = self.loader.getNextImage()
            classes_string = ','.join(classes)
            tFile.write(image_path+'\t'+classes_string+'\n')
            # Compute descriptors
            des = self._createDescriptors(im)
            if des != None:
                # Store descriptors
                des_list.append((image_path, des))
            else:
                empty_feature_list.append(image_path)
        self.loader.closeIteration()

        # Cluster descriptors to create "vocabulary" of visual words
        # over image training set
        self.vocab, variance = self._clusterVisualWords(des_list, self.vocab_size)

        # Calculate histogram of features for each image using defined vocabulary
        # and calculated image descriptors
        self._im_features = np.zeros((len(des_list),self.vocab_size), "float32")
        for i in range(len(des_list)):
            self._im_features[i] = self._createHistOfFeatures(des_list[i][1])

        # Standardize feature set
        self._im_features = self._standardize(self._im_features)

        # Associate image path with image's feature histogram in dictionary
        # so that it can be looked up later
        for i in range(len(self._im_features)):
            self.feat_dict[des_list[i][0]] = self._im_features[i]

        # Add zero entries for images where no features could be detected
        for i in range(len(empty_feature_list)):
            self.feat_dict[empty_feature_list[i]] = np.zeros((self.vocab_size,))

        #print 'dict keys:\n' + str(self.feat_dict.keys())


    def getFeatureSet(self):
        return self._im_features

    # Retrieve the histogram of features for the parameter image
    # which is from the training set. Not to be confused with the extractImageFeatures() method.
    def getFeaturesForThisImage(self,image_path):
        #for key, _ in self.feat_dict.iteritems():
        #    print "key: " + str(key) + "\n"
        return self.feat_dict[image_path]

    # Compute unstandardized feature histogram for a new, previously unseen image
    # based on vocabulary of visual words defined over training image set
    def extractImageFeatures(self, img):
        #print 'getting test image features...'
        #des = self._createDescriptors(img)
        _, des = self.surf.detectAndCompute(img,None)
        hist = self._createHistOfFeatures(des)
        return hist

    # Extract descriptors from parameter image
    def _createDescriptors(self, img):
        # Detect keypoints
        #kpts = self.fea_det.detect(img)
        
        # Create descriptors from keypoints
        #_, des = self.des_ext.compute(img, kpts)
        _, des = self.surf.detectAndCompute(img, None)
        return des

    # Perform k-means clustering on descriptor set of training image set
    def _clusterVisualWords(self, des_list, k):
        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]
        descriptor_matrix = np.zeros((instances, self.vocab_size))
        i = 0
        for _, descriptor in des_list[1:]:
            #print str(len(descriptor))
            #print 'currently at: ' + str(i)
            i = i + 1
            if descriptor != None:
                #descriptor_matrix[i][:] = descriptor
                descriptors = np.vstack((descriptors, descriptor))
        # vocab is the vocabulary of visual "words" or descriptors
        print "descriptors.shape: " + str(descriptors.shape)
        vocab, variance = kmeans(descriptors, k, 1)
        return vocab, variance

    # Create histogram of features that will represent
    # the parameter image and will be used for classification.
    # This is done by assigning each descriptor in parameter "descriptors" its nearest
    # visual "word" in the vocabulary as defined in parameter vocab
    def _createHistOfFeatures(self, descriptors):
        feats = np.zeros((1, self.vocab_size), "float32")
        # Each descriptor in the descriptor list is assigned its nearest visual "word".
        # words is a length M array, where M is the number of descriptors for
        # the given image. Each entry in words stores an index to the nearest
        # visual word in the vocabulary.
        words, distance = vq(descriptors,self.vocab)
        # for each vocabulary index in words, increment the count for that word
        # in the histogram
        for w in words:
            feats[0][w] += 1
        return feats


    # Perform Tf-Idf vectorization
    # see (https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for full explanation
    # Basically term frequency (tf) and inverse document frequency (idf) are weighting factors that scale the
    # words in the bag of features vocabulary based on their frequency in the training data set.
    def tfIdf(feats, des_list):
        nbr_occurences = np.sum( (feats > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')
        return idf

    # Standardize features: center by removing mean and scaling to unit variance
    def _standardize(self,vect):
        # Standardize features: center by removing mean and scaling to unit variance
        stdSlr = StandardScaler().fit(vect)
        return stdSlr.transform(vect)

def _test():
    # Path to image data set
    datapath = 'C:\\Users\\Wim\\Documents\\AIDKE\\Project 1\\Data set\\foodimages\\foodimages'
    # Size of vocabulary of visual features/words
    vocab_size = 500
    # Create image loader to be passed to BoF instance
    loader = ImageLoader('C:\Users\Wim\Documents\AIDKE\Project 1\New Code\image_classification.csv',datapath)
    
    # Create BoF instance
    bof = BoF(loader,vocab_size)
    # Create bag-of-features
    bof.createBagOfWords()

    # Path to test image
    # test_datapath = 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\foodimages\\foodimages\\pp1\\26.11.2013 13_47_00.jpg'
    loader.startIteration()
    # Load the image
    [img,classes,image_path] = loader.getNextImage()
    loader.closeIteration()
    # Extract features from test image according to defined vocabulary
    feats = bof.extractImageFeatures(img)
    print 
    # Print extracted features
    print 'features for image:\n' + str(bof.getFeaturesForThisImage('C:\\Users\\Wim\\Documents\\AIDKE\\Project 1\\Data set\\foodimages\\foodimages\\pp1\\25.11.2013 11_14_29.jpg'))


def main():
    _test()

if __name__ == '__main__':
    main()