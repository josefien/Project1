import numpy as np
import cv2 as cv

# Enum voor classes schrijven
# Excel file inlezen en classes extraheren
# Gebruik s.strip().split(',') om lines met commas in aparte classes te scheiden
# Maak een string van alle classes met een spatie ertussen
# Maak een Enum van deze string als volgt:
# Foods = Enum( 'Foods', category_string)

# Wat moeten we kunnen doen?
# Images inlezen en bewerkingen doen, zoals:
# feature extraction: hier ben ik wat meer mee bekend door de presentatie, kijk hier eens naar om te beginnen, dan kom je vast al een hoop dingen tegen.
# classification

kernel_size = 20 # size of gabor filter
sigma = 5 # standard deviation of Gaussian envelope
gamma = 1 # aspect ratio
psi = -90*np.pi/180 # phase offset

class GaborFilter:
    # Parameter information from
    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#getgaborkernel

    # Create Gabor filter with parameter wavelengths and orientations
    def __init__(self, labdas, thetas):
        print('creating object')
        self.kernels = GaborFilter.createGaborKernels(labdas, thetas)

    # Need to create set of Gabor kernels once and store it
    @staticmethod
    def createGaborKernels(labdas, thetas):
        print('at createGaborKernels')
        kernels = [None] * len(labdas) * len(thetas)
        k = 0
        for i in xrange(len(labdas)):
            for j in xrange(len(thetas)):
                kernels[k] = cv.getGaborKernel((kernel_size,kernel_size),sigma,thetas[j],labdas[i],gamma,psi)
                k += 1
        return kernels

    # Get Gabor texture information from input image
    def getGaborFeatures(self,image):
        # Divide image into number of blocks according to parameter block_size
        numOfRows = 1
        numOfColumns = 1
        rowHeight = image.shape[1]/numOfRows
        colWidth = image.shape[0]/numOfColumns

        print('rowHeight: ',rowHeight)
        print('colWidth: ',colWidth)

        for i in xrange(numOfRows):
            for j in xrange(numOfColumns):
                # Convolve each block b with each Gabor kernel i to get response matrix b_i
                for kernel in self.kernels:
                    response = cv.filter2D(image[j*colWidth:(j+1)*colWidth,i*rowHeight:(i+1)*rowHeight], cv.CV_32F,kernel)
                    # Show image
                    cv.imshow('Response', response)
                    cv.waitKey(0)
                    #print('response.shape:',response.shape)
                    # Create one statistic for each b_i
                    #E_i = np.matrix(np.square(response)).sum() # Local energy of response matrix
                    #print('Energy of matrix',E_i)

        # Local Energy = summing up the squared value of each matrix value from a response matrix
        # Mean Amplitude = sum of absolute values of each matrix value from a response matrix
        # Maybe check some other statistics as well

if __name__ == '__main__':
    #labda = 21
    #theta = 0
    labdas = [11, 16, 21]
    thetas = [0, 45, 90]
    img = cv.imread('banana.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img.shape)
    filt = GaborFilter(labdas, thetas)
    filt.getGaborFeatures(img)