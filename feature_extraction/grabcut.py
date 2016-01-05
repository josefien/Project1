import numpy as np
import cv2


def generateMask(img):
    height, width = img.shape[:2]
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (int(width*0.05),int(height*0.05),int(width*0.9),int(height*0.9))
    
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    return np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
def modifyImage(img):
    mask = generateMask(img)
    return img*mask[:,:,np.newaxis]
    
    