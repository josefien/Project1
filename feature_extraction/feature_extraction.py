import sys
sys.path.append('../util')
from image_loader import *
import cv2
import numpy as np
import random
import BoF as bf
import gabor_filter as gf
import decimal

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
def bof_init(loader):
    # Size of vocabulary of visual features/words
    # Try 500 to start with
    vocab_size = 100
    surf_threshold = 4500
    
    # Create BoF instance
    bof = bf.BoF(loader,vocab_size,surf_threshold)
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

def apply_combined(to_extract,classpath,bof_model):
    # Create feature vector
    # Apply gabor filter
    gb_vector = []
    if 'gabor' in to_extract:
        gb_list = gabor(classpath)
        gb_vector = np.asarray(gb_list)
     
    # Apply BoF filter
    bf_vector = []
    if 'bof' in to_extract:
        bf_vector = np.asarray(apply_bof(classpath,bof_model))
        
    # Calculate histogram
    hist_vector = []
    if 'hist' in to_extract:
        hist_vector = histogram(classpath)

    # Create image feature vector by concatenating all vectors
    feature_vector = np.concatenate((gb_vector, bf_vector,hist_vector))
    return feature_vector

if __name__ == '__main__':

    datasets = ['dataset0','dataset2']
    csv_names = ['balanced_200.csv']
    feature_methods = ['bof']
    for dataset in datasets:
        for csv_name in csv_names:
            for to_extract in feature_methods:
                path_to_images = 'C:/Users/Wim/Documents/AIDKE/Project 1/Data set/foodimages/' + dataset + '/'
                loader = ImageLoader('../' + csv_name + '.csv',path_to_images)
                # Which features to extract, all is 
                # ['hist','bof','gabor']
                f_prefix = "_".join(to_extract)
                # Sub directory in feature-directory to write feature-files to
                directory = dataset

                # Initialize the BoF model
                bof_model = 0
                if 'bof' in to_extract:
                    bof_model = bof_init(loader)
                    # File of the paths of images used for training the BoF model
                    tFile = open('bof_trainset.txt','r')

                path = 'C:/Users/Wim/Documents/AIDKE/Project 1/New Code/feature_extraction/'
                # File to which the feature vectors are written to
                fFile = open(path + directory + '/' + f_prefix + '_' + dataset + '_' + csv_name + '_features.txt', 'w')
                # File to which the classes are written to
                cFile = open(path + directory + '/' + f_prefix + '_' + dataset + '_' + csv_name + '_classes.txt','w')
                
                all_features = []
                all_classes = []

                i = 0

                if 'bof' in to_extract:
                    for line in tFile:
                        if i%1000 == 0:
                            print('Processing image no. {}'.format(i))
                        i = i + 1
                        info = line.split('\t')
                        classpath = info[0]
                        classes_string = info[1]
                        classpath_s = classpath.replace(path_to_images,'',1)
                        classes_string = classpath_s + '\t' + classes_string 
                        all_classes.append(classes_string)
                        feature_vector = apply_combined(to_extract,classpath,bof_model)
                        all_features.append(feature_vector)

                if not 'bof' in to_extract:
                    loader.startIteration()
                    while loader.hasNext():
                        if i%1000 == 0:
                            print('Processing image no. {}'.format(i))
                        i = i + 1    
                        [img, classes, classpath] = loader.getNextImage() 
                        classes_string = ','.join(classes)
                        classpath_s = classpath.replace('C:/Users/Wim/Documents/AIDKE/Project 1/','',1) 
                        classes_string = classpath_s + '\t' + classes_string + '\n'
                        all_classes.append(classes_string)
                        feature_vector = apply_combined(to_extract,classpath,bof_model)
                        all_features.append(feature_vector)
                    loader.closeIteration()
                
                feature_matrix = np.matrix(np.array(all_features))
                np.savetxt(fFile,feature_matrix)
                for i in range(len(all_classes)):
                    cFile.write(all_classes[i])
                print("Done!")
