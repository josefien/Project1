from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal


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
    # Try 500 to start with
    vocab_size = 50
    
    # Create BoF instance
    bof = bf.BoF(loader,vocab_size)
    # Create bag-of-features
    bof.createBagOfWords()
    return bof

# Histogram
def histogram(classpath):
    # Histogram size
    bin = 64
    img = cv2.imread(classpath)
    blue = cv2.calcHist([img],[0],None,[bin],[0,256])
    green = cv2.calcHist([img],[0],None,[bin],[0,256])
    red = cv2.calcHist([img],[0],None,[bin],[0,256])
    red = (np.asarray(red).reshape(-1))
    green = (np.asarray(green).reshape(-1))
    blue = (np.asarray(blue).reshape(-1))
    return np.concatenate((blue,green,red))
    #return result

def apply_bof(classpath,bof_model):
    feats = bof_model.getFeaturesForThisImage(classpath)
    return feats

# Total of 680 images in database
if __name__ == '__main__':
    bof_model = bof_init()
    #n = 10
    # File to which the feature vectors are written to
    fFile = open('features.txt', 'w')
    # File to which the classes are written to
    cFile = open('classes.txt','w')
    
    all_features = []
    all_classes = []
    loader.startIteration()
    #for i in range(n):	
    while loader.hasNext():		
        [img, classes, classpath] = loader.getNextImage() 
        classes_string = ','.join(classes)
        all_classes.append(classes_string)
       
        print(classpath)
        # create feature vectors
        # apply gabor filter
        gb_list = gabor(classpath)
        gb_vector = np.asarray(gb_list)
      
        # apply BoF filter
        bf_vector = apply_bof(classpath,bof_model)
        
        # Calculate histogram
        hist_vector = histogram(classpath)
       
        # create image feature vector by appending all feature vectors
        feature_vector = np.concatenate((gb_vector, bf_vector,hist_vector))

        all_features.append(feature_vector)

    loader.closeIteration()
    feature_matrix = np.matrix(np.array(all_features))
    np.savetxt(fFile,feature_matrix)
    for i in range(len(all_classes)):
        cFile.write(all_classes[i] + '\n')
    #cFile.write(all_classes[-1])