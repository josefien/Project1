from image_loader import *




loader = ImageLoader('../image_classification.csv','C:/Users/Nadine/Documents/University/Uni 2015/RPMAI1/foodimages/foodimages')

loader.startIteration()
for i in range(10):			#while loader.hasNext():
	[img, classes,classpath] = loader.getNextImage()
	cv2.imshow(', '.join(classes),img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
loader.closeIteration()

