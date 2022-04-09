import os
import cv2
import cells
import numpy as np
import collections
import pandas as pd
import seaborn as sns
from heapq import nlargest
from typing import OrderedDict 
import matplotlib.pyplot as plt



class Tissue:
    
    def __init__(self,cellNetwork):
        self.cellNetwork = cellNetwork
        self.workingMemory = cellNetwork.workingMemory



    def getFeedback(self):
        '''
        Atoms:     How we process   :  def processEyes(self, k=4):
        Molecules: Threshold of DPR :  def getDpr(self, threshold=75)
        Cells:     Depth of K       :  def knn(self, k=8)
        '''

        newBelief = collections.defaultdict(int)
        feelingSpectrum = nlargest(2, self.workingMemory, key = self.workingMemory.get)

        self.f1 = feelingSpectrum[0]
        self.f1Value = self.workingMemory.get(self.f1) 
        if len(feelingSpectrum)>1: 
            self.f2 = feelingSpectrum[1]
            self.f2Value = self.workingMemory.get(self.f2)  
        else:
            self.f2 = 0
            self.f2Value = 0


        self.feelingPercent = self.f1Value + self.f2Value
        # print('Feeling Polarity:')
        # print('main feeling:')
        # print(self.f1,'accounts for ', self.f1Value)
        # print('')
        # print(self.f2, 'accounts for ', self.f2Value)              
        print(self.f1, 'and', self.f2, 'make up', self.f1Value + self.f2Value)


        


        # Heuristic Approach 
        # TODO: GET ALL COMBOS OF 2
        # fear/surprise
        # joy/anger
        # joy/netural 
        # joy/surprise
        # sadness/anger
        # etc

        #Systems Approach
        # Too generalized
        # Too overtrained
        # Just right
        # No idea :(
