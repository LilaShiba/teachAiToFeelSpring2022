import os
from unicodedata import name
import cv2
import heapq
import collections
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps 
from PIL import ImageChops
from PIL import ImageFilter
import matplotlib.pyplot as mlt

class knn:
    def __init__(self,) -> None:
        
        
class detectEyes:
   
    def __init__(self, filePath):
        img = cv2.imread(filePath)
        img = self.crop_bottom_half(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(img,1.03, 2)
        
        # print(eyes)
        # limit to top half of face brah
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        cv2.imshow('bork',img)
        cv2.waitKey(0)
                    
        
        
        
        
    def crop_bottom_half(self,img):
        # https://stackoverflow.com/questions/44759654/divide-image-into-two-equal-parts-python-opencv/44764659
        height, width, channels = img.shape
        #croppedImage = img[height:int(height/2), 0:width] #this line crops
        start_row, start_col = int(0), int(0)
        # Let's get the ending pixel coordinates (bottom right of cropped top)
        end_row, end_col = int(height * .8), int(width)
        cropped_top = img[start_row:end_row , start_col:end_col]
        #cropped_img = img[img.shape[0]/2:img.shape[0]]
        # cv2.imshow("cropped", cropped_top)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cropped_top


if __name__ == "__main__":
    # for label in os.listdir('images/train'):
    #     for img in 
    for label in os.listdir('images/train'):
        for img in os.listdir('images/train/'+label):
            deltaPath = 'images/train/'+label+'/'+img 
            detectEyes(deltaPath)
        #print(trainingSet.graph)