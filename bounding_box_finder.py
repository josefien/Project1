import cv2
from image_loader import *
import numpy as np
import sys

contours_returned = 5

class BoundingBoxFinder:

	@staticmethod
	def get_bounding_box(img):
		# Preprocess image
		edges = BoundingBoxFinder._preprocess(img)
		# Detect contours
		contours = BoundingBoxFinder._find_contours(edges, contours_returned)

		# Make a copy of the original image that we can draw on
		BoundingBoxFinder._draw_contours(img,contours)
		BoundingBoxFinder._draw_bounding_box(img,contours)


	@staticmethod
	def _find_contours(edges, num_contours):
		(contours, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:num_contours]
		return contours

	# Preprocess image for contour finding by converting to gray scale
	# and performing edge detection.
	@staticmethod
	def _preprocess(img):
		blur = cv2.bilateralFilter(img,9,75,75)
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150)
		cv2.imshow('edges',edges)
		return edges

	@staticmethod
	def _draw_contours(img,contours):
		drawing_copy = img.copy()
		cv2.drawContours(drawing_copy, contours, -1, (0,255,0), 3)
		cv2.imshow('image',drawing_copy)
		cv2.waitKey(0)

	@staticmethod
	def _draw_bounding_box(img,contours):
		# print 'len(contours) = ' + str(len(contours))
		# print 'len(contours[0]) = ' + str(len(contours[0]))
		# print 'len(contours[0][0]) = ' + str(len(contours[0][0]))
		# print 'len(contours[0][0][0]) = ' + str(len(contours[0][0][0]))
		# print 'len(contours[0][0][0][0]) = ' + str(contours[0][0][0][0])
		minx = sys.maxint
		miny = sys.maxint
		maxx = -sys.maxint
		maxy = -sys.maxint

		for i in range(len(contours)):
			for j in range(len(contours[i])):
				# print 'contours[i][0][j][0] = ' + str(contours[i][j][0][0])
				# print 'contours[i][0][j][1] = ' + str(contours[i][j][0][1])
				minx = min(minx, contours[i][j][0][0])
				maxx = max(maxx, contours[i][j][0][0])
				miny = min(miny, contours[i][j][0][1])
				maxy = max(maxy, contours[i][j][0][1])

		# print 'minx = ' + str(minx)
		# print 'miny = ' + str(miny)
		# print 'maxx = ' + str(maxx)
		# print 'maxy = ' + str(maxy)

		#print 'type(contours[0][0]) = ' + str(type(contours[0][0][0]))
		#print 'contours[0][0][0] = ' + str(contours[0][0][0])
		drawing_copy = img.copy()
		cv2.rectangle(drawing_copy,(minx,maxy),(maxx,miny),(0,255,0),2)
		cv2.imshow('bounding_box',drawing_copy)
		cv2.waitKey(0)


def _test():
	num_of_images = 10
	datapath = 'C:\Users\Wim\Documents\AIDKE\Project 1\Data set\\foodimages\\foodimages'
	loader = ImageLoader('image_classification.csv',datapath)
	loader.startIteration()
	# Load images and perform bounding box operation
	for i in xrange(num_of_images):
		[img,_,_] = loader.getNextImage()
		BoundingBoxFinder.get_bounding_box(img)
	loader.closeIteration()


def __main__():
	_test()

if __name__ == '__main__':
	__main__()