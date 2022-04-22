import os
import cv2 
import dlib
import collections
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# memoryPath = label
# 
# set D => { 
#                   label:[ (0)...(N+1)],
#                    ...
#                }

# Build = 0:1 build new dataset

class Filter:

    def __init__(self,memoryPath,build=1):
        # takes in new whole new dataset to
        # filter on
        self.memoryPath = memoryPath
        self.build = build

    def filterSubRoutine(self,imgPath):
        '''
            Input:
            imgPath = 'emotionLabel/imgFolderPath'


            Outputs: 
            self.rect => the left and right eye balls
            self.coords => location (x,y) of left and right eye balls
            self.leftEyeCords  / self.rightEyeCords => location (x,y) of left and right eye balls
            self.leftEyeArray / self.rightEyeArrayv => numpy array of eye images
            self.leftEyeImg  /  self.rightEyeImg => eye cropped to face-overlap parameter numpy array
        
        '''
        

        self.path = imgPath
        self.gray  = cv2.imread(imgPath)
        #self.img  = cv2.imread(imgPath)
        #self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        detector  = dlib.get_frontal_face_detector()
        self.rects = detector(self.gray,1)

        # If two eyes in photo; continue processing
        if len(self.rects) > 1:
            self.proccessImg()
        else:
        # shame them for potatoe
            print('quality of image is shit. Please try harder')

    def proccessImg(self):
        predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        for (i, rect) in enumerate(self.rects):
          
            self.shape = predictor(self.gray, rect)
            shape = self.getEyes()

    def getEyes(self):
        leftEye  = [36,37,38,39,40,41]
        rightEye = [42,43,44,45,46,47]
        if self.shape:
            lEye, rEye = [], []
            coords = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                coords[i] = (self.shape.part(i).x, self.shape.part(i).y)
              
            self.leftEyeCords =  {'x1': coords[36][0], 'x2': coords[39][0], 'y1':coords[38][1], 'y2': coords[41][1]}
            self.rightEyeCords = {'x1': coords[42][0], 'x2': coords[45][0], 'y1':coords[43][1], 'y2': coords[46][1]}
        
            self.coords = coords
            self.processEyes()
            
    def processEyes(self):
        # cropped = img[start_row:end_row, start_col:end_col]
        k = self.threshold
        img = self.img
        #crop_img = img[y:y+h, x:x+w]
        # We only want data with two eyes for basic training
        if self.leftEyeCords and self.rightEyeCords:
            self.leftEyeImg = img[self.leftEyeCords['y1']-k: self.leftEyeCords['y2']+k, 
                                    self.leftEyeCords['x1']-k:self.leftEyeCords['x2']+k].copy()
            
            self.rightEyeImg = img[self.rightEyeCords['y1']-k: self.rightEyeCords['y2']+k, 
                                    self.rightEyeCords['x1']-k:self.rightEyeCords['x2']+k].copy()

            self.leftEyeArray  =  np.array(self.leftEyeImg.copy())
            self.rightEyeArray =  np.array(self.rightEyeImg.copy())

    def getDpr(self,threshold=25):
        left  = self.leftEyeArray.copy()
        right = self.rightEyeArray.copy()
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
       

    def createFolder(self,emotionFolder,iteration):
        newPath = emotionFolder+'/'+iteration
        os.makedirs(newPath)
        lEye = np.array(self.leftFilterImg)
        rEye = np.array(self.rightFilterImg)
        # Filter Img
        cv2.imwrite(newPath+'/leftEyeFilter.png',lEye)
        cv2.imwrite(newPath+'_'+iteration+'/rightEyeFilter.png',rEye)
        cv2.imwrite(newPath+'_'+iteration+'/LeftEye.png', self.leftEyeImg)
        cv2.imwrite(newPath+'_'+iteration+'/RightEye.png', self.rightEyeImg)




if __name__ == '__main__':
    dataSet = 'dataSetOne'
    deltaDataSet = collections.defaultdict(list)
    ds = Filter(dataSet)

    for label in os.listdir(dataSet):
        print(label)
        iteration = 0
        # per each label    
        for imgPath in os.listdir(dataSet+'/'+label):
            # per each image
            delta = label+'/'+imgPath
            ds.filterSubRoutine(delta)
            ds.getDpr()
            ds.createFolder(delta, iteration)
            iteration += 1





    # def percieve(self,memoryPath='eyeData', k=2):
    
    #     graph = collections.defaultdict(list)
    #     knnMap = collections.defaultdict(list)
        
    #     for label in os.listdir(memoryPath):    
    #         for imgFolder in os.listdir(memoryPath+label):
    #             # if imgFolder in ['neutral']:
    #             #     continue
    #             deltaPath = memoryPath+label+'/'+imgFolder
    #             #if len(deltaPath) >= 5:
    #             delta = Molecule(label, deltaPath,self.dprThreshold,self.deltaPath)
    #                 # both eyes y'all
    #             if delta.x and delta.y:
    #                 x = round(delta.x,k)
    #                 y = round(delta.y,k)
    #                 if abs(x-y) <= 25:
    #                 #if 1 == 1:
    #                     # z = round(delta.z,2)
    #                     graph[ (x,y) ].append(delta.label)
    #                     cords = (x,y)
    #                     #cords[ (x,y,z) ].append(delta.vibe)
    #                     knnMap[(x,y)].append((delta.label, delta, cords))

    #     self.graph = graph 
    #     #self.cords = cords
    #     self.knnMap = knnMap

    #     res = pd.DataFrame(graph.items())
    #     res = res.rename(columns={0: "cords", 1:'emotion'})
    #     res['x'], res['y'] = zip(*res["cords"])
        

    #     mapOfEmotions = pd.DataFrame()
    #     for idx,row in res.iterrows():
    #         #if len(row['emotion']) >= 3:
    #         vote = Counter(row['emotion'])
    #         #TODO make dynamic
    #         # change .50 for vote can make dynamic
    #         if vote.most_common(1)[0][1] / len(row['emotion']) > .50:
    #             row['emotion'] = vote.most_common(1)[0][0]
    #             mapOfEmotions = mapOfEmotions.append(row)
    #     # cleanMap = pd.DataFrame.from_dict(mapOfEmotions)
    #     # print(cleanMap)

    #     self.mapOfEmotions = mapOfEmotions
