from image_loader import *
import cv2
import numpy as np

n = 10

# Return 2 lists: one list of images, one list of labels
# For now, just 10 images + labels
def load_base():
    loader = ImageLoader('image_classification.csv','C:/Users/Nadine/Documents/University/Uni 2015/RPMAI1/foodimages/foodimages')
    loader.startIteration()
    images = []
    all_classes = []
    for i in range(n):			#while loader.hasNext():
        [img, classes] = loader.getNextImage() 
        #classes_string = ', '.join(classes)
        images.append(img)
        all_classes.append(classes)
        #cv2.imshow(', '.join(classes),img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    loader.closeIteration()
    return images, all_classes

# Total of 680 images in database
# For now: simple division, first 90% of images for train, rest for test
# Write to files
#def train_test():



#def write_labels_to_numbers():



if __name__ == '__main__':
	train = open('train_set', 'w')
	test = open('test_set','w')
	[img,classes] = load_base()
	j=0
	for i,c in zip(img,classes):
		if(j<0.9*n):
			print "train"
		if(j>0.9*n):
			print "test"
		j = j + 1
		print j
		print 0.9*n

