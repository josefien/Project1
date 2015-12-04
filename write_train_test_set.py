from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf


loader = ImageLoader('image_classification.csv','C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\foodimages\\foodimages')

# Gabor filter taken from Wim's implementation
def gabor(classpath):
    labdas = [11, 16, 21]
    thetas = [0, 45, 90]
    numrows = 3
    numcols = 3
    img = cv2.imread(classpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filt = gf.GaborFilter(labdas, thetas, numrows, numcols)
    vect = filt.getGaborFeatures(img)
    return vect

# BoF filter taken from Wim's implementation
def bof_init():
    # Size of vocabulary of visual features/words
    vocab_size = 50
    
    # Create BoF instance
    bof = bf.BoF(loader,vocab_size)
    # Create bag-of-features
    bof.createBagOfWords()
    return bof

def apply_bof(classpath,bof_model):
    feats = bof_model.getFeaturesForThisImage(classpath)
    return feats

def convert_matrix_to_vector(matrix):
    (m,n) = matrix.shape
    print m
    print n


# For now: simple division, first 90% of images for train, rest for test
# Total of 680 images in database
if __name__ == '__main__':
    bof_model = bof_init()
    n = 10
    train = open('train_set', 'w')
    test = open('test_set','w')

    loader.startIteration()
    images = []
    all_classes = []
    for i in range(n):			#while loader.hasNext():
        [img, classes, classpath] = loader.getNextImage() 
        classes_string = ', '.join(classes)
        images.append(img)
        all_classes.append(classes)
        
        # create feature vectors
        # apply gabor filter
        gb_vector = gabor(classpath)
    
        # apply BoF filter
        bf_matrix = apply_bof(classpath,bof_model)

        convert_matrix_to_vector(bf_matrix)
        
        new = gb_vector.append(bf_vector)
        #print(new)
        #print(bf_vector)
        # create image feature vector by appending all feature vectors

        # write feature vector + label to train or test file
    loader.closeIteration()



