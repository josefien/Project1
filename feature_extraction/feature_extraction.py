import sys
sys.path.append('C:/Users/Nadine/git/Project1/util')
from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal

path_to_images = 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\foodimages\\grabcut'
loader = ImageLoader('../image_classification.csv',path_to_images)

# Gabor filter taken from Wim's implementation
def gabor(classpath):
    #lambdas = wavelengths, try some values in the range [10,20]
    #thetas = directions, try values in the range [0,180]
    labdas = [11, 16, 21]
    thetas = [0, 45, 90]
    # Into how many sections the image will be divided
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
    # Try 500 to start with
    vocab_size = 500
    
    # Create BoF instance
    bof = bf.BoF(loader,vocab_size)
    # Create bag-of-features
    bof.createBagOfWords()
    loader.closeIteration()
    return bof

# Histogram
def histogram(classpath):
    # Histogram size
    bin = 64
    img = cv2.imread(classpath)
    channels = cv2.split(img)
    features = []
    for chan in channels:
        hist = cv2.calcHist([chan],[0],None,[bin],[1,256])
        hist = hist/np.sum(hist)
        hist = np.asarray(hist).reshape(-1)
        features = np.concatenate((features,hist))
    return features

def apply_bof(classpath,bof_model):
    feats = bof_model.getFeaturesForThisImage(classpath)
    #feats = extractImageFeatures(img)
    return feats

if __name__ == '__main__':
    # Initialize the BoF model
    bof_model = bof_init()
    
    # File to which the feature vectors are written to
    fFile = open('features.txt', 'w')
    # File to which the classes are written to
    cFile = open('classes.txt','w')
    # File of the paths of images used for training the BoF model
    tFile = open('bof_trainset.txt','r')

    all_features = []
    all_classes = []
    #for i in range(100):   
    for line in tFile:
        #[img, classes, classpath] = newLoader.getNextImage() 
        info = line.split('\t')
        classpath = info[0]
        classes_string = info[1]
        #classes_string = ','.join(classes)
        
        classpath_s = classpath.replace(path_to_images,'',1)
        classes_string = classpath_s + '\t' + classes_string 
        #classes_string = classpath_s + '\t' + classes_string + '\n'
        all_classes.append(classes_string)
       
        # Create feature vector
        # Apply gabor filter
        gb_list = gabor(classpath)
        gb_vector = np.asarray(gb_list)
      
        # Apply BoF filter
        bf_vector = np.asarray(apply_bof(classpath,bof_model))
        
        # Calculate histogram
        hist_vector = histogram(classpath)

        # Create image feature vector by appending all vectors
        feature_vector = np.concatenate((gb_vector, bf_vector,hist_vector))
        #feature_vector = np.concatenate((gb_vector, hist_vector))
        all_features.append(feature_vector)

    loader.closeIteration()
    feature_matrix = np.matrix(np.array(all_features))
    np.savetxt(fFile,feature_matrix)
    for i in range(len(all_classes)):
        cFile.write(all_classes[i])
    #cFile.write(all_classes[-1])
    print("Done!")