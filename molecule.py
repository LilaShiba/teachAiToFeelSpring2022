
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
        self.leftEye, self.rightEye = cv2.imread(left), cv2.imread(right) 
        self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
        self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
        self.leftArray, self.rightArray = np.array(self.leftEye), np.array(self.rightEye)
        # For KNN Plot
        # x = left eye dpr
        self.x = None
        self.y = None 


    def getDpr(self):
        pass 

    def blurToGaus(self):
        pass

    def showEyes(self):
        pass

    def useScore(self):
        pass 


