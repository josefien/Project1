import cv2
from image_loader import *

contours_returned = 5

class BoundingBoxFinder:

	@staticmethod
	def get_bounding_box(img):
		# Preprocess image
		edges = BoundingBoxFinder._preprocess(img)
		# Detect contours
		contours = BoundingBoxFinder._find_contours(edges, contours_returned)

		print 'len(contours) = ' + str(len(contours))

		# Make a copy of the original image that we can draw on
		drawing_copy = img.copy()
		cv2.drawContours(drawing_copy, contours, -1, (0,255,0), 3)
		cv2.imshow('image',drawing_copy)
		cv2.waitKey(0)


	@staticmethod
	def _find_contours(edges, num_contours):
		(contours, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:num_contours]
		return contours

	# Preprocess image for contour finding by converting to gray scale
	# and performing edge detection.
	@staticmethod
	def _preprocess(img):
		blur = cv2.bilateralFilter(img,9,150,75)
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 75)
		cv2.imshow('edges',edges)
		return edges

def _test():
	print 'in test method'
	datapath = 'C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages'
	loader = ImageLoader('image_classification.csv',datapath)
	loader.startIteration()
	# Load images and perform bounding box operation
	for i in xrange(5):
		[img,_,_] = loader.getNextImage()
		BoundingBoxFinder.get_bounding_box(img)
	loader.closeIteration()


def __main__():
	_test()

if __name__ == '__main__':
	__main__()