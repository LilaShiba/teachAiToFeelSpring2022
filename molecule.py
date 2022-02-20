
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
        self.leftEye, self.rightEye = cv2.imgread(left), cv2.imgread(right) 
        self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
        self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
        self.leftArray, self.rightArray = np.array(self.leftEye), np.array(self.rightEye)
        # For KNN Plot
        self.x = None
        self.y = None 
