import collections
import os
import cv2 
from PIL import Image
from cells import Cell
from atoms import Atom 
from tissue import Tissue
from molecules import Molecule
import matplotlib.pyplot as plt


class graphInput():
    
    def __init__(self,label,imgPath,iteration,feedback):
        self.feedback = feedback
        self.label = label 
        self.imgPath = imgPath
        self.iteration =  iteration
        faceOverlap = feedback['faceOverlap']
        dprThreshold = feedback['dprThreshold']
        knnDepth = feedback['knnDepth']
    
        # Process Face Data (DPR)
        # on init,  will run
        atom = Atom(label, imgPath,iteration,faceOverlap=4) 
        atom.createMolecule(label)

        molecule = Molecule(label, atom.moleculeImgPath, dprThreshold=100)
        print('x:',  molecule.x)
        print('y:',  molecule.y)
        # knn graph where 
        # xAxis=dprRightEye, yAxis=dprLeftEye , hue=label
        molecule.train()
        # cool stuff > molecule.showMap(), print(molecule.knnMap)
        # Analogus Reasoning
        cellNetwork = Cell(molecule.x, molecule.y, molecule.mapOfEmotions, knnDepth=7)
        cellNetwork.knn(5)
        # distro of feelings for working memory, e.g., result of cell.knn aggregated 
        cellNetwork.gatherAnalogiesView()
        # Systems Brah aka tissue
        tissue = Tissue(cellNetwork)
        tissue.getFeedback()

    def processFeedback(self):
        lvl = self.tissue.feelingPercent
        if 70 > lvl  >=40:
            #lower k in knn -> nearest neighbors should be majority of feeling
            self.feedback['knnDepth'] -= 1 
        
        if lvl < 40:
            self.feedback['knnDepth'] += 1 
            #rasie k in knn -> more context is needed for situation
         
        



if __name__ == '__main__':
    label = 'predict'
    iteration = 0
    # TODO: create folder to hold each iteration's mental map to look for somekind of intelligence

    feedback = {'faceOverlap':4, 'dprThreshold':100, 'knnDepth':8}
    while iteration < 3:
        prediction = graphInput(label, '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/39843011-angry-face-man.webp',iteration,feedback)
        feedback = prediction.feedback
        iteration+=1
