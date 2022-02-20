
import os 
import cv2
import collections
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

emotionVibes = {
                'fear':1,
                'anger':2,
                'disgust':3,
                'joy': 4,
                'surprise': 5,
                'neutral': 6,
                'sadness': 7,
                }

class Molecule:

    def __init__(self,label,filePath):
        self.label = label 
        self.filePath = filePath
        self.x = None
        self.y = None
        self.vibe = emotionVibes[label]

        
        if len(os.listdir(filePath)) > 1:
            left,right  = os.listdir(filePath)
            # Basic Processing
            self.leftEye, self.rightEye = cv2.imread(filePath+'/'+left), cv2.imread(filePath+'/'+right) 
            self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
            self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
            # self.leftArray, self.rightArray = np.array(self.leftEyeGrey), np.array(self.rightEyeGrey)
            
            self.leftArray = self.blurToGaus(self.leftEyeGrey)
            self.rightArray = self.blurToGaus(self.rightEyeGrey)
            self.getDpr()
           
           

    def getDpr(self):
        left  = self.leftArray.copy()
        right = self.rightArray.copy()
        # Left EYE
        left[left < 51] = 1
        left[left > 1]  = 0
        dpcLeft = np.count_nonzero(left) 
        self.dprLeftEye = (dpcLeft / 480 )
        # RIGHT EYE
        right[right < 51] = 1
        right[right > 1]  = 0
        dpcRight = np.count_nonzero(right) 
        
        self.dprRightEye = (dpcRight / 480 )

        # print('ldpr', self.dprLeftEye)
        # print('rdpr', self.dprRightEye)
        self.x = self.dprRightEye
        self.y = self.dprLeftEye

    def blurToGaus(self, imgToBlur, kernal=(0,0)):
        # resource: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
        # GAUSS
        # deltaImg = cv2.GaussianBlur(imgToBlur,kernal,cv2.BORDER_DEFAULT)
        deltaImg = cv2.bilateralFilter(imgToBlur,5,5,5)
        # cv2.imshow('bilateral filter', deltaImg)
        # cv2.waitKey(0) # waits until a key is pressed
        # cv2.destroyAllWindows() 
        return np.array(deltaImg)

    def showEyes(self):
        pass

    def useScore(self):
        pass 





if __name__ == '__main__':
    graph = collections.defaultdict(list)
    cords = collections.defaultdict(list)
    for label in os.listdir('eyeData'):
        
        for imgFolder in os.listdir('eyeData/'+label):
            deltaPath = 'eyeData/'+label+'/'+imgFolder
            if len(deltaPath) > 1:
                delta = Molecule(label, deltaPath)
                if delta.x and delta.y:
                    x = round(delta.x,3)
                    y = round(delta.y,3)
                    graph[ (x,y) ].append(delta.label)
                    cords[ (x,y) ] = delta.vibe
    
    
    

                #cols x,y,label
    print('processing dpr done')
    res = pd.DataFrame(cords.items())
    # print(res)
    # res = res.rename(columns={0: "cords", 1:'emotion'})
    # res['x'], res['y'] = zip(*res["cords"])
    # print('scatter')
    # sns.scatterplot(data=res, x='x', y='y')
    # plt.show()
    
    res = res.rename(columns={0: "cords", 1:'emotion'})
    res['x'], res['y'] = zip(*res["cords"])
    print('scatter')
    print(res)
    sns.scatterplot(data=res, x='x', y='y', hue='emotion')
    plt.show()
