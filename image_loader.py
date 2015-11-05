import numpy as np
import os
import cv2
import re


##	ImageLoader provides an iteration through the image-database
#	matches images with its labels
#	@author corth

class ImageLoader:

	__table = None	#file-object
	__img = None	#next image to provide
	__classes = []	#classes/categories concerning the next image
	
	##	Constructor
	#	tablePath - path to csv-table with lables
	#	imagePath - path to image-directory
	def __init__(self, tablePath, imagePath):
		self.__tp = tablePath
		if not imagePath.endswith(os.path.sep):
			self.__ip = imagePath + os.path.sep
		else:
			self.__ip = imagePath
	
	##	Start your iteration (don't forget get close)
	# 	may block your table-file
	def startIteration(self):
		self.closeIteration();
		self.__table = open(self.__tp, 'r')
		line = self.__table.readline()
		self.__loadNextImage();
		
	##	returns if there is at least one image left
	#	used for iterations -> while hasNext ...
	def hasNext(self):
		return self.__img is not None
		
	##	get the next image 
	def getNextImage(self):
		image = self.__img
		category = self.__classes
		self.__loadNextImage();
		return image, category
			 
	##	close the current table
	def closeIteration(self):
		if not self.__table is None:
			self.__table.close()
		

	##	inner function to buffer next image and categories
	def __loadNextImage(self):
		self.__img = None	# reset next image
		self.__classes = None
		line = self.__table.readline()
		while line:	#look for a valid line
			line = re.sub("\n|\"|'","",line).split(',') # split row in cells (in csv devided by ','
			path = self.__ip+line[0]+os.path.sep	# concatenated imagepath from table
			for filename in os.listdir(path ):	# is there a matching file?
				if line[1].replace(':','_') in filename:
					self.__img = cv2.imread(path+filename)	# matching file found
					self.__classes = line[3].split(' ')
					return
			line = self.__table.readline()	# no image found for this line -> go on with next line

