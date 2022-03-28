import os
from typing import DefaultDict, OrderedDict 
import cv2
import collections
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import molecules

class Cell:
    def __init__(self,x,y,mapOfWorld):
        self.x = x
        self.y = y
        self.target = (x,y)
        self.mapOfWorld = mapOfWorld
        self.edges = None # pd.DataFrame()
        self.connectionStrength = collections.defaultdict(list)

    def knn(self, k):
        mOw = self.mapOfWorld
        #mOw = pd.DataFrame(mOw.items())
        #mOw = mOw.rename(columns={0: "idx", 1:'molecule'})
        mOw['dist'] = mOw['cords'].apply(self.dist_heuristic)
        #mOw['x'], mOw['y'] = zip(*mOw["idx"])
        mOw = mOw.sort_values(by=['dist'], ascending=True)
        self.edges = mOw[0:k]
        # print('self.edges:')
        # print(self.edges)

    def dist_heuristic(self, cords):
        xa, ya = self.target
        xb, yb = cords
        return np.sqrt((xa-xb)**2 + (ya-yb)**2)

        

    def gatherAnalogiesView(self):
        # TODO vectorize
        '''
        returns scatter plot with prediction shown as red circle
        ''' 
        # collection of edges
        self.workingMemory = collections.defaultdict(int)
        count = 0
            
        for idx, row in self.edges.iterrows():
            emotion = row[1]
            dist = row[4]
            self.workingMemory[emotion] += 1
            count += 1
            
        for key, item in self.workingMemory.items():
            self.workingMemory[key] = item / count
        print(self.workingMemory)
        print('wow')

            


            #showing images

            # files = os.listdir(molecule.filePath)
            # deltaPathL = molecule.filePath +'/'+files[0] 
            # deltaPathR = molecule.filePath +'/'+files[1] 


            # print('edge left eye:')
            # print('deltaPathL:', deltaPathL)
            # l = Image.open(deltaPathL)
            # l.show()
            
            # print('edge right eye:')
            # print('deltaPathR:', deltaPathR)
            # r = Image.open(deltaPathR)
            # r.show()
            # break  
       

    def gatherAggregations(self):
        '''
          Show distrubution of feelings per cord, rather than vote
          Returns self.distributions[(cords)] = histogram/dict of 
          feelings and percentage of overall feeling.  
        '''
        pass

    def createTissue(self):
        '''
            return network of analogus cells
        '''
        pass