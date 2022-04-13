import collections
import os
import cv2 
from PIL import Image
from cells import Cell
from atoms import Atom 
from tissue import Tissue
from molecules import Molecule
import matplotlib.pyplot as plt


thoughtProcess = collections.defaultdict()
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
        atom = Atom(label,imgPath,iteration,faceOverlap) 
        atom.createMolecule(label)
        if len(atom.imgPath) > 1:
            molecule = Molecule(label, atom.moleculeImgPath,dprThreshold,atom.delta)
            # TODO: Implement shortTermMemory x,y cords history
            print('x:',  molecule.x)
            print('y:',  molecule.y)
            # knn graph where 
            # xAxis=dprRightEye, yAxis=dprLeftEye , hue=label
            molecule.train()
            
            # cool stuff > molecule.showMap(), print(molecule.knnMap)
            # Analogus Reasoning
            cellNetwork = Cell(molecule.x, molecule.y, molecule.mapOfEmotions, knnDepth)
            cellNetwork.knn(knnDepth)
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
        
        
        # Heuristic -> Not filtering pixels correctly 
        # if lvl < 0.70 and self.feedback['dprThreshold'] < 100:
        #     self.feedback['dprThreshold'] += 25
        #     #rasie k in knn -> more context is needed for situation
        if lvl >= .80:
            return 0
        #     return self.feedback
        
        if lvl < .70 and self.feedback['faceOverlap'] < 4:
            self.feedback['faceOverlap'] += 1
            return self.feedback

        
        # Heuristic -> Detecting Pixels, too much noise 
        if lvl < 0.80:
            if self.feedback['knnDepth'] > 3:
                self.feedback['knnDepth'] -= 2
            if self.feedback['dprThreshold']< 125:
                self.feedback['dprThreshold'] += 25
            
        elif lvl < 0.60:
            self.feedback['knnDepth'] += 3

        thoughtProcess[self.iteration] = self.feedback
        return self.feedback




        

         
        



if __name__ == '__main__':
    label = 'predict'
    iteration = 0
    # TODO: create folder to hold each iteration's mental map to look for somekind of intelligence

    feedback = {'faceOverlap':1, 'dprThreshold':10, 'knnDepth':7}
    shortTermMemory = []
    while iteration < 4:
        print('feedback:', feedback)
        prediction = graphInput(label, '/Users/kylejames/Desktop/teachAiToFeelSpring2022-main/angerTest.jpeg',iteration,feedback)
        feedback = prediction.processFeedback()
        if feedback == 0:
            break
        iteration+=1
    print(thoughtProcess)




