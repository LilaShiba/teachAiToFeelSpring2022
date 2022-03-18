import os
from typing import OrderedDict 
import cv2
import collections
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Cell:
    def __init__(self,x,y,mapOfWorld):
        self.x = x
        self.y = y
        self.target = (x,y)
        self.mapOfWorld = mapOfWorld

    def knn(self, k):
        mOw = self.mapOfWorld
        mOw = pd.DataFrame(mOw.items())
        mOw = mOw.rename(columns={0: "idx", 1:'molecule'})
        mOw['dist'] = mOw['idx'].apply(self.dist_heuristic)
        mOw = mOw.sort_values(by=['dist'], ascending=True)
        print(mOw.head(k))

    def dist_heuristic(self, cords):
        xa, ya = self.target
        xb, yb = cords
        return np.sqrt((xa-xb)**2 + (ya-yb)**2)

        

    def gatherAnalogiesView(self, target):
        '''
        returns scatter plot with prediction shown as red circle
        '''
        # out put
        # Sort by distance
        df = {'cords':(target.x, target.y),'emotion':'predict','x':target.x,'y':target.y}
        targetDataFrame = pd.DataFrame(df)
        #print(targetDataFrame.iloc[-1])
        target.mapOfEmotions.append(targetDataFrame, ignore_index=True)   
        sns.scatterplot(data=target.mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep",label='DPR Rate Right and Left Eye')
        plt.scatter(x=target.x, y=target.y, color='r', s=30)
        plt.legend()
        plt.show()

    def createAnalogies(self, target, k=5):
        '''
            target.knnMap => {(cords):(label, analogousMolecule)}
            returns subregion of knnMap (aka network of cells => tissue), knnMapSubRegion, with links to K analogous molecules
        '''
        pass

    def createTissue(self):
        '''
            return network of analogus cells
        '''
        pass
