import os
from typing import OrderedDict 
import cv2
import collections
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cells


class Tissue:
    
    def __init__(self,cellNetwork):
        self.cellNetwork = cellNetwork
        self.workingMemory = cellNetwork.workingMemory



    def transportNutrients(self):
        '''
        Atoms:     How we process   :  def processEyes(self, k=4):
        Molecules: Threshold of DPR :  def getDpr(self, threshold=75)
        Cells:     Depth of K       :  def knn(self, k)
        '''
        pass     