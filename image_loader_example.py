from image_loader import *




loader = ImageLoader('image_classification.csv','<<your path to image-directory>>')

loader.startIteration()
for i in range(10):			#while loader.hasNext():
	[img, classes] = loader.getNextImage()
	cv2.imshow(', '.join(classes),img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
loader.closeIteration()

