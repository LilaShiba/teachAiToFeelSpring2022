import os
from typing import OrderedDict 
import cv2
import collections
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Cell:
    def __init__(self,target):
        self.x = target.x
        self.y = target.y
        self.target = target

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
        plt.scatter(x=target.x, y=target.y, color='r', s=40)
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
