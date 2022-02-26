# Heuristic Approach
import os 
import cv2
import collections
from collections import Counter
import numpy as np 
import pandas as pd
import seaborn as sns
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
            self.leftArray, self.rightArray = np.array(self.leftEyeGrey), np.array(self.rightEyeGrey)
            #self.leftArray = self.blurToGaus(self.leftEyeGrey)
            #self.rightArray = self.blurToGaus(self.rightEyeGrey)
            self.getDpr()
           
           
    def getDpr(self, threshold=75):
        left  = self.leftArray.copy()
        right = self.rightArray.copy()
        # Left EYE
        left[left < threshold] = 1
        left[left > 1]  = 0
        dpcLeft = np.count_nonzero(left) 
        zero_countL = np.count_nonzero(left==0)#left[np.where(left == 0)]
        self.dprLeftEye = (dpcLeft / (zero_countL+dpcLeft) )
        
        # RIGHT EYE
        right[right < threshold] = 1
        right[right > 1]  = 0
        zero_countR = np.count_nonzero(right==0)#right[np.where(right == 0)]
        dpcRight = np.count_nonzero(right) 
        self.dprRightEye = (dpcRight / (dpcRight + zero_countR) )
        
        self.x = self.dprLeftEye#abs(zero_countL-zero_countR) #self.dprRightEye#self.dprRightEye
        self.y = self.dprRightEye#abs(self.dprRightEye-self.dprLeftEye)#self.dprLeftEye#zero_countR #self.dprLeftEye
        self.z = abs(dpcLeft-dpcRight)#round(self.dprRightEye,2)#abs(dpcLeft-dpcRight)


    def blurToGaus(self, imgToBlur, kernal=(0,0)):
        # resource: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
        # GAUSS
        # deltaImg = cv2.GaussianBlur(imgToBlur,kernal,cv2.BORDER_DEFAULT)
        deltaImg = cv2.bilateralFilter(imgToBlur,0,0,0)
        # cv2.imshow('bilateral filter', deltaImg)
        # cv2.waitKey(0) # waits until a key is pressed
        # cv2.destroyAllWindows() 
        return np.array(deltaImg)

    def showEyes(self):
        pass

    def useScore(self):
        return self.z  

    def train(self):
        colors = {
                0:'orange',
                1:'yellow',
                2:'purple',
                3: 'green',
                4: 'blue',
                5: 'black',
                6: 'gold',
                'predict': 'red'
                }
        
        graph = collections.defaultdict(list)
        cords = collections.defaultdict(list)
        
        for label in os.listdir('eyeData'):    
            for imgFolder in os.listdir('eyeData/'+label):
                # if imgFolder in ['neutral']:
                #     continue
                deltaPath = 'eyeData/'+label+'/'+imgFolder
                if len(deltaPath) > 1:
                    delta = Molecule(label, deltaPath)
                    # both eyes y'all
                    if delta.x and delta.y:
                        x = round(delta.x,1)
                        y = round(delta.y,1)
                        z = round(delta.z,1)
                        graph[ (x,y) ].append(delta.label)
                        cords[ (x,y,z) ].append(delta.vibe)
        self.graph = graph 
        self.cords = cords

    def predict(self):
        pass
            # sns.scatterplot(data=mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
            # plt.show()
        







                #cols x,y,label
    # print('processing dpr done')
    # res = pd.DataFrame(graph.items())
    # #print(res)
    # res = res.rename(columns={0: "cords", 1:'emotion'})
    # # Z score for usablity
    # #res['x'], res['y'], res['z'] = zip(*res["cords"])
    # res['x'], res['y'] = zip(*res["cords"])
    
    # res = res.sort_values('x')
    # # emotion score -> avg value of cord arrays
    # #res[sum(res['emotion'])]
    # #res[res['z'] > 10] = -1
    
    # # mapOfEmotions = pd.DataFrame(columns=['x','y','z','emotion'])
    # # print(mapOfEmotions)

    # mapOfEmotions = pd.DataFrame()
    # for idx,row in res.iterrows():
    #     if row['emotion'] != -1 and len(row['emotion']) > 1:
    #         vote = Counter(row['emotion'])
    #         if vote.most_common(1)[0][1] > 1:
    #             row['emotion'] = vote.most_common(1)[0][0]
    #             mapOfEmotions = mapOfEmotions.append(row)
    # # cleanMap = pd.DataFrame.from_dict(mapOfEmotions)
    # # print(cleanMap)
    # print(mapOfEmotions)


    # plt.bar(res.keys(), res.values(), 1, color='g')
    # for x in res.iterrows():
    #     print(x)

    # sns.scatterplot(data=mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
    # plt.show()









# Layered Approach: Give range of values rather than single one
# import os 
# import cv2
# import collections
# from collections import Counter
# import numpy as np 
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# emotionVibes = {
#                 'fear':0,
#                 'anger':1,
#                 'disgust':2,
#                 'joy': 3,
#                 'surprise': 4,
#                 'neutral': 5,
#                 'sadness': 6,
#                 }

# class Molecule:

#     def __init__(self,label,filePath):
#         self.label = label 
#         self.filePath = filePath
#         self.x = None
#         self.y = None
#         self.vibe = emotionVibes[label]

        
#         if len(os.listdir(filePath)) > 1:
#             left,right  = os.listdir(filePath)
#             # Basic Processing
#             self.leftEye, self.rightEye = cv2.imread(filePath+'/'+left), cv2.imread(filePath+'/'+right) 
#             self.leftEyeGrey  = cv2.cvtColor(self.leftEye, cv2.COLOR_BGR2GRAY)
#             self.rightEyeGrey = cv2.cvtColor(self.rightEye, cv2.COLOR_BGR2GRAY)
#             self.leftArray, self.rightArray = np.array(self.leftEyeGrey), np.array(self.rightEyeGrey)
#             #self.leftArray = self.blurToGaus(self.leftEyeGrey)
#             #self.rightArray = self.blurToGaus(self.rightEyeGrey)
#             self.getDpr()
           
           

#     def getDpr(self, threshold=75):
#         left  = self.leftArray.copy()
#         right = self.rightArray.copy()
#         # Left EYE
#         left[left < threshold] = 1
#         left[left > 1]  = 0
#         dpcLeft = np.count_nonzero(left) 
#         zero_countL = np.count_nonzero(left==0)#left[np.where(left == 0)]
#         self.dprLeftEye = (dpcLeft / (zero_countL+dpcLeft) )
        
#         # RIGHT EYE
#         right[right < threshold] = 1
#         right[right > 1]  = 0
#         zero_countR = np.count_nonzero(right==0)#right[np.where(right == 0)]
#         dpcRight = np.count_nonzero(right) 
#         self.dprRightEye = (dpcRight / (dpcRight + zero_countR) )
        
#         self.x = self.dprLeftEye#abs(zero_countL-zero_countR) #self.dprRightEye#self.dprRightEye
#         self.y = self.dprRightEye#abs(self.dprRightEye-self.dprLeftEye)#self.dprLeftEye#zero_countR #self.dprLeftEye
#         self.z = abs(dpcLeft-dpcRight)#round(self.dprRightEye,2)#abs(dpcLeft-dpcRight)


#     def blurToGaus(self, imgToBlur, kernal=(0,0)):
#         # resource: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
#         # GAUSS
#         # deltaImg = cv2.GaussianBlur(imgToBlur,kernal,cv2.BORDER_DEFAULT)
#         deltaImg = cv2.bilateralFilter(imgToBlur,0,0,0)
#         # cv2.imshow('bilateral filter', deltaImg)
#         # cv2.waitKey(0) # waits until a key is pressed
#         # cv2.destroyAllWindows() 
#         return np.array(deltaImg)

#     def showEyes(self):
#         pass

#     def useScore(self):
#         pass 





# if __name__ == '__main__':
#     graph = collections.defaultdict(list)
#     cords = collections.defaultdict(list)
#     for label in os.listdir('eyeData'):
        
#         for imgFolder in os.listdir('eyeData/'+label):
#             # if imgFolder in ['neutral']:
#             #     continue
#             deltaPath = 'eyeData/'+label+'/'+imgFolder
#             if len(deltaPath) > 1:
#                 delta = Molecule(label, deltaPath)
#                 if delta.x and delta.y:
#                     x = round(delta.x,1)
#                     y = round(delta.y,1)
#                     z = round(delta.z,1)
#                     graph[ (x,y) ].append(delta.label)
#                     cords[ (x,y,z) ].append(delta.vibe)
    
    
#     colors = {
#                 0:'red',
#                 1:'yellow',
#                 2:'purple',
#                 3: 'green',
#                 4: 'blue',
#                 5: 'black',
#                 6: 'gold'
#                 }

#                 #cols x,y,label
#     print('processing dpr done')
#     res = pd.DataFrame(cords.items())
#     #print(res)
#     res = res.rename(columns={0: "cords", 1:'emotion'})
#     # Z score for usablity
#     res['x'], res['y'], res['z'] = zip(*res["cords"])
#     res = res.sort_values('x')
#     # emotion score -> avg value of cord arrays
#     #res[sum(res['emotion'])]
#     res[res['z'] > 10] = -1
    
#     # mapOfEmotions = pd.DataFrame(columns=['x','y','z','emotion'])
#     # print(mapOfEmotions)

#     mapOfEmotions = pd.DataFrame()
#     for idx,row in res.iterrows():
#         if row['emotion'] != -1 and len(row['emotion']) > 1:
#             vote = Counter(row['emotion'])
#             if vote.most_common(1)[0][1] > 1:
#                 row['emotion'] = vote.most_common(1)[0][0]
#                 mapOfEmotions = mapOfEmotions.append(row)
#     # cleanMap = pd.DataFrame.from_dict(mapOfEmotions)
#     # print(cleanMap)
#     print(mapOfEmotions)


#     # plt.bar(res.keys(), res.values(), 1, color='g')
#     # for x in res.iterrows():
#     #     print(x)

#     sns.scatterplot(data=mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
#     plt.show()

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')     
    # ax.scatter(mapOfEmotions['x'], mapOfEmotions['y'], mapOfEmotions['z'], c=mapOfEmotions['emotion'].map(colors))
    # plt.show()
