import cv2
import numpy as np
from matplotlib import pyplot as plt
from gabor_filter import *

def _surftest():
	img = cv2.imread('C:\Users\Wim\Documents\AIDKE\\banana2.jpg')
	blur = cv2.bilateralFilter(img,10,100,100)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	surf = cv2.SURF(400)
	surf.hessianThreshold = 750
	kp, des = surf.detectAndCompute(blur,None)
	
	img2 = cv2.drawKeypoints(gray,kp,None,[0,255,0],4)
	cv2.imshow('keypoints',img2)
	cv2.imwrite('banana2_surf2.jpg',img2)
	cv2.waitKey(0)

def _colortest():
	img = cv2.imread('C:\Users\Wim\Documents\AIDKE\\banana2.jpg')
	h = np.zeros((300,256,3))
	b,g,r = img[:,:,0],img[:,:,1],img[:,:,2]
	bins = np.arange(257)
	bin = bins[0:-1]
	color = [ (255,0,0),(0,255,0),(0,0,255) ]

	for item,col in zip([b,g,r],color):
	    N,bins = np.histogram(item,bins)
	    v=N.max()
	    N = np.int32(np.around((N*255)/v))
	    N=N.reshape(256,1)
	    pts = np.column_stack((bin,N))
	    cv2.polylines(h,[pts],False,col,2)

	h=np.flipud(h)
	cv2.imshow('img',h)
	cv2.imwrite('banana2_color.jpg',h)
	cv2.waitKey(0)

def _altcolortest():
	img = cv2.imread('C:\Users\Wim\Documents\AIDKE\\banana2.jpg')
	chans = cv2.split(img)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title("'Flattened' Color Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	features = []
	 
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
	 
		# plot the histogram
		plt.plot(hist, color = color)
		plt.xlim([0, 256])

def _gabortest():
	img = cv2.imread('C:\Users\Wim\Documents\AIDKE\\banana2.jpg')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	filt = GaborFilter([11, 16, 21], [0, 45, 90], 1, 1)
	filt.getGaborFeatures(gray)

def __main__():
	_surftest()

if __name__ == '__main__':
	__main__()