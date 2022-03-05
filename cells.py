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

    def gatherAnalogies(self, target):
        # out put
        # Sort by distance
        df = {'cords':(target.x, target.y),'emotion':'predict','x':target.x,'y':target.y}
        targetDataFrame = pd.DataFrame(df)
        #print(targetDataFrame.iloc[-1])
        target.mapOfEmotions.append(targetDataFrame, ignore_index=True)   
        sns.scatterplot(data=target.mapOfEmotions, x='x', y='y', hue='emotion',style='emotion',palette="deep")
        plt.scatter(x=target.x, y=target.y, color='r')
        plt.legend()
        plt.show()
       