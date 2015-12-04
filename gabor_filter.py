import numpy as np
import cv2 as cv

kernel_size = 20 # size of gabor filter
sigma = 5 # standard deviation of Gaussian envelope
gamma = 1 # aspect ratio
psi = -90*np.pi/180 # phase offset

class GaborFilter:
    # Parameter information from
    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#getgaborkernel

    # Create Gabor filter with parameter wavelengths and orientations
    def __init__(self, labdas, thetas, numrows, numcols):
        print('creating object')
        self.__createGaborKernels(labdas, thetas)
        self.__numOfRows = numrows
        self.__numOfCols = numcols

    # Create set of Gabor kernels once and store it
    def __createGaborKernels(self, labdas, thetas):
        self.__kernels = [None] * len(labdas) * len(thetas)
        k = 0
        for i in xrange(len(labdas)):
            for j in xrange(len(thetas)):
                self.__kernels[k] = cv.getGaborKernel((kernel_size,kernel_size),sigma,thetas[j],labdas[i],gamma,psi)
                k += 1

    # Get Gabor texture information from input image
    def getGaborFeatures(self,image):
        # Divide image into number of blocks according to parameter block_size
        rowHeight = image.shape[1]/self.__numOfRows
        colWidth = image.shape[0]/self.__numOfCols

        #print('rowHeight: ',rowHeight)
        #print('colWidth: ',colWidth)

        # Vectors into which we will store Gabor filter information
        # for each channel (each combination of orientation and wavelength)
        # and for each block of the image
        energyVector = [];
        amplitudeVector = [];

        for i in xrange(self.__numOfRows):
            for j in xrange(self.__numOfCols):
                # Convolve each block b with each Gabor kernel i to get response matrix b_i
                for kernel in self.__kernels:
                    response = cv.filter2D(image[j*colWidth:(j+1)*colWidth,i*rowHeight:(i+1)*rowHeight], cv.CV_32F,kernel)
                    # Store information in vectors
                    energyVector.append(getLocalEnergy(response))
                    amplitudeVector.append(getMeanAmplitude(response))
                    # Show image
                    #cv.imshow('Response', response)
                    #cv.waitKey(0)
        return energyVector + amplitudeVector


# Compute local energy of parameter response matrix.
# Local Energy = sum of squared value of each element in the passed response matrix
def getLocalEnergy(response):
    return np.matrix(np.square(response)).sum()

# Compute mean amplitude of parameter response matrix.
# Mean Amplitude = sum of absolute value of each element in the passed response matrix
def getMeanAmplitude(response):
    return np.matrix(np.absolute(response)).sum()

def __test__():
    labdas = [11, 16, 21]
    thetas = [0, 45, 90]
    numrows = 3
    numcols = 3
    img = cv.imread('banana.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img.shape)
    filt = GaborFilter(labdas, thetas, numrows, numcols)
    vect = filt.getGaborFeatures(img)
    print vect[:20]
    

if __name__ == '__main__':
    __test__()