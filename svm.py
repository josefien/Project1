from image_loader import *
import cv2
import numpy as np

# Return 2 lists: one list of images, one list of labels
# For now, just 10 images + labels
def load_base():
    loader = ImageLoader('image_classification.csv','C:/Users/Nadine/Documents/University/Uni 2015/RPMAI1/foodimages/foodimages')
    loader.startIteration()
    images = []
    all_classes = []
    for i in range(10):			#while loader.hasNext():
        [img, classes] = loader.getNextImage() 
        #classes_string = ', '.join(classes)
        images.append(img)
        all_classes.append(classes)
        #cv2.imshow(', '.join(classes),img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    loader.closeIteration()
    return images, all_classes

# Load images

# For now: first n are for training, remaining m images for testing

# Write labels to numbers

# Get features, for now: dummy feature extraction method

# Transform to the right format (rows: samples, columns, features + labels (in seperate vector))

# Train SVM

# Predict SVM


#
class StatModel(object):  
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

# SVM Wrapper class
class SVM(StatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        #return self.model.predict_all(samples).ravel()
        return np.float32( [self.model.predict(s) for s in samples])


# Histogram of Oriented Gradients, taken from opencv python tutorials
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

if __name__ == '__main__':
    imgs,classes = load_base()
    print(classes)

    for img in imgs:
        print(img)
    samples = np.array(np.random.random((10,2)), dtype = np.float32)
    print(samples)
    y_train = np.array([2.,0.,0.,2.,0.,1.,0.,2.,1.,2.], dtype = np.float32)

    clf = SVM()
    clf.train(samples, y_train)
    y_val = clf.predict(samples)
    print(y_val)