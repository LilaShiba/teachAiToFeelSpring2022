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
        self.f2 = feelingSpectrum[1]
        self.f2Value = self.workingMemory.get(self.f2)   

        print('Feeling Polarity:')
        print('main feeling:')
        print(self.f1,'accounts for ', self.f1Value)
        print('')
        print(self.f2, 'accounts for ', self.f2Value)              
        print('these two feeling make up', self.f1Value + self.f2Value)


    def processFeedback(self):
        pass
        # if 70% >main vote >=40%:
            # lower k in knn -> nearest neighbors should be majority of feeling
        
        # if main vote < 40%:
            # rasie k in knn -> more context is needed for situation
        


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
