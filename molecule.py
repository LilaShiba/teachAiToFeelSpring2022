
import os 
import cv2
import numpy as np 

class Molecule:

    def __init__(self,label,filePath):
        self.label = label 
        self.filePath = filePath
        # [leftEyeImg,rightEyeImg]
        left,right = os.listdir(filePath)
        # Basic Processing
        self.leftEye, self.rightEye = cv2.imread(filePath+'/'+left), cv2.imread(filePath+'/'+right) 
        self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
        self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
        self.leftArray, self.rightArray = np.array(self.leftEyeGrey), np.array(self.rightEyeGrey)
        # For KNN Plot
        # x = left eye dpr
        self.x = None
        self.y = None 


    def getDpr(self):
        left  = self.leftArray.copy()
        right = self.rightArray.copy()
        # Left EYE
        left[left < 105] = 1
        left[left > 1]  = 0
        dpcLeft = np.count_nonzero(left) 
        self.dprLeftEye = (dpcLeft / 480 )
        print('ldpr', self.dprLeftEye)
        # RIGHT EYE
        right[right < 105] = 1
        right[right > 1]  = 0
        dpcRight = np.count_nonzero(right) 
        print('dpc', dpcRight, 'pixels', 480 )
        self.dprRightEye = (dpcRight / 480 )





    def blurToGaus(self, imgToBlur, kernal=(0,0)):
        # resource: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
        # GAUSS
        # deltaImg = cv2.GaussianBlur(imgToBlur,kernal,cv2.BORDER_DEFAULT)
        deltaImg = cv2.bilateralFilter(imgToBlur,2,2,2)
        cv2.imshow('bilateral filter', deltaImg)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() 
        return deltaImg

    def showEyes(self):
        pass

    def useScore(self):
        pass 





if __name__ == '__main__':
    path = 'eyeData/surprise/4'
    m1 = Molecule('surprise', path)
    # m1.blurToGaus(m1.leftEyeGrey)
    m1.getDpr()