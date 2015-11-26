from image_loader import *
import cv2
import numpy as np
import random
import BoF as bof
import gabor_filter as gf

n = 10
loader = ImageLoader('image_classification.csv','C:/Users/Nadine/Documents/University/Uni 2015/RPMAI1/foodimages/foodimages')
train = open('train_set', 'w')
test = open('test_set','w')

def gabor(image):
    labdas = [11, 16, 21]
    thetas = [0, 45, 90]
    numrows = 3
    numcols = 3
    img = cv.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img.shape)
    filt = GaborFilter(labdas, thetas, numrows, numcols)
    vect = filt.getGaborFeatures(img)
    print vect[:20]
    
# For now: simple division, first 90% of images for train, rest for test
# Total of 680 images in database
if __name__ == '__main__':
    loader.startIteration()
    images = []
    all_classes = []
    for i in range(n):			#while loader.hasNext():
        [img, classes, classpath] = loader.getNextImage() 
        classes_string = ', '.join(classes)
        gabor(img)
        images.append(img)
        all_classes.append(classes)
        r = random.random()
        if(r<0.9):
            train.write(img)
        else:
            test.write(img)

        #cv2.imshow(', '.join(classes),img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    loader.closeIteration()



