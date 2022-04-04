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

        molecule = Molecule(label, atom.moleculeImgPath, dprThreshold)
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
        self.atom = atom 
        self.molecule = molecule 
        self.cells = cellNetwork
        self.tissue = tissue

    def processFeedback(self):
        lvl = self.tissue.feelingPercent
        print('lvl:',lvl)
        if lvl >=40 and lvl < 85:
            #lower k in knn -> nearest neighbors should be majority of feeling
            self.feedback['knnDepth'] -= 2
            #self.feedback['dprThreshold'] += 25
            return self.feedback
        
        if lvl < 40:
            self.feedback['dprThreshold'] += 25
            #rasie k in knn -> more context is needed for situation
            return self.feedback
         
        



if __name__ == '__main__':
    label = 'predict'
    iteration = 0
    # TODO: create folder to hold each iteration's mental map to look for somekind of intelligence

    feedback = {'faceOverlap':4, 'dprThreshold':50, 'knnDepth':8}
    while iteration < 3:
        print('feedback:', feedback)
        prediction = graphInput(label, 'images/selfTest.jpg',iteration,feedback)
        feedback = prediction.processFeedback()
        iteration+=1
