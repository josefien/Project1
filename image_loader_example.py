from image_loader import *




loader = ImageLoader('image_classification.csv','C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages')

loader.startIteration()
for i in range(10):			#while loader.hasNext():
	[img, classes, _] = loader.getNextImage()
	cv2.imshow(', '.join(classes),img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
loader.closeIteration()

