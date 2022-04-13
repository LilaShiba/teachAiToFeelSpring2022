# Heuristic Approach
import os 
import cv2
import numpy as np 
import collections
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt


emotionVibes = {
                'fear':0,
                'anger':1,
                'disgust':2,
                'joy': 3,
                'surprise': 4,
                'neutral': 5,
                'sadness': 6,
                'predict':-1
                }

class Molecule:

    def __init__(self,label,filePath, dprThreshold, delta):
        self.label = label 
        self.filePath = filePath
        self.x = None
        self.y = None
        self.vibe = emotionVibes[label]
        self.graph = collections.defaultdict(list)
        self.dprThreshold = dprThreshold
        self.deltaPath = delta

        
        if len(os.listdir(filePath)) > 1:
            left,right  = os.listdir(filePath)
            # Basic Processing
            self.leftEye, self.rightEye = cv2.imread(filePath+'/'+left), cv2.imread(filePath+'/'+right) 
            self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
            self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
            self.leftArray, self.rightArray = np.array(self.leftEyeGrey), np.array(self.rightEyeGrey)
            #self.leftArray = self.blurToGaus(self.leftEyeGrey)
            #self.rightArray = self.blurToGaus(self.rightEyeGrey)
            self.getDpr()

    def createFolder(self):
        lEye = np.array(self.leftFilterImg)
        rEye = np.array(self.rightFilterImg)
        cv2.imwrite(self.deltaPath+'/leftEyeFilter.png',lEye)
        cv2.imwrite(self.deltaPath+'/rightEyeFilter.png',rEye)
        
                   
    def getDpr(self):
        threshold = self.dprThreshold
        left  = self.leftArray.copy()
        right = self.rightArray.copy()
        # Left EYE
        left[left <= threshold] = 1
        left[left > 1]  = 0
        
        lpcL = np.count_nonzero(left==1) 
        dpcL = np.count_nonzero(left==0)#left[np.where(left == 0)]
        self.dprLeftEye = (lpcL / (lpcL+dpcL) )
        left[ left > 0] = 255
        self.leftFilterImg = left
        
        # RIGHT EYE
        right[right <= threshold] = 1
        right[right > 1]  = 0
        dpcR = np.count_nonzero(right==1)
        lpcR = np.count_nonzero(right==0) 
       

        self.dprRightEye = (dpcR / (dpcR + lpcR) )
        right[ right > 0] = 255
        self.rightFilterImg = right
        self.x = self.dprLeftEye#abs(zero_countL-zero_countR) #self.dprRightEye#self.dprRightEye
        self.y = self.dprRightEye#abs(self.dprRightEye-self.dprLeftEye)#self.dprLeftEye#zero_countR #self.dprLeftEye
        #self.z = abs(dpcLeft-dpcRight)#round(self.dprRightEye,2)#abs(dpcLeft-dpcRight)
        if self.label == 'predict':
            self.createFolder()


    def blurToGaus(self, imgToBlur, kernal=(0,0)):
        # resource: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
        # GAUSS
        # deltaImg = cv2.GaussianBlur(imgToBlur,kernal,cv2.BORDER_DEFAULT)
        deltaImg = cv2.bilateralFilter(imgToBlur,0,0,0)
        # cv2.imshow('bilateral filter', deltaImg)
        # cv2.waitKey(0) # waits until a key is pressed
        # cv2.destroyAllWindows() 
        return np.array(deltaImg)

    def show(self):
        map = sns.scatterplot(data=self.mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
        plt.legend()
        plt.show()

    def useScore(self):
        return self.z  

    def train(self, k=2):
    
        colors = {
                0:'teal',
                1:'yellow',
                2:'purple',
                3: 'green',
                4: 'blue',
                5: 'black',
                6: 'gold',
                -1:'red'
                }
        
        graph = collections.defaultdict(list)
        #cords = collections.defaultdict(list)
        knnMap = collections.defaultdict(list)
        
        for label in os.listdir('eyeData'):    
            for imgFolder in os.listdir('eyeData/'+label):
                # if imgFolder in ['neutral']:
                #     continue
                deltaPath = 'eyeData/'+label+'/'+imgFolder
                #if len(deltaPath) >= 5:
                delta = Molecule(label, deltaPath,self.dprThreshold,self.deltaPath)
                    # both eyes y'all
                if delta.x and delta.y:
                    x = round(delta.x,k)
                    y = round(delta.y,k)
                    if abs(x-y) <= 25:
                    #if 1 == 1:
                        # z = round(delta.z,2)
                        graph[ (x,y) ].append(delta.label)
                        cords = (x,y)
                        #cords[ (x,y,z) ].append(delta.vibe)
                        knnMap[(x,y)].append((delta.label, delta, cords))

        self.graph = graph 
        #self.cords = cords
        self.knnMap = knnMap

        res = pd.DataFrame(graph.items())
        res = res.rename(columns={0: "cords", 1:'emotion'})
        res['x'], res['y'] = zip(*res["cords"])
        

        mapOfEmotions = pd.DataFrame()
        for idx,row in res.iterrows():
            #if len(row['emotion']) >= 3:
            vote = Counter(row['emotion'])
            if vote.most_common(1)[0][1] / len(row['emotion']) > .50:
                row['emotion'] = vote.most_common(1)[0][0]
                mapOfEmotions = mapOfEmotions.append(row)
        # cleanMap = pd.DataFrame.from_dict(mapOfEmotions)
        # print(cleanMap)

        self.mapOfEmotions = mapOfEmotions


        #print('map of emotions:', mapOfEmotions)
        
    def showMap(self):
        sns.scatterplot(data=self.mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
        plt.plot(self.x, self.y, marker="o", markersize=15, color="red")
        plt.legend()
        plt.show()

    def predict(self):
        pass