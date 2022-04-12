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
        #     return self.feedback
        
        # Heuristic -> Detecting Pixels, too much noise 
        if self.feedback['dprThreshold'] >= 50 and lvl < 0.70:
            if self.feedback['knnDepth'] > 2:
                self.feedback['knnDepth'] -= 2
            else:
                self.feedback['dprThreshold'] += 25
            return self.feedback
        elif lvl > 0.70:
            self.feedback['knnDepth'] += 3

        return self.feedback


        

         
        



if __name__ == '__main__':
    label = 'predict'
    iteration = 0
    # TODO: create folder to hold each iteration's mental map to look for somekind of intelligence

    feedback = {'faceOverlap':4, 'dprThreshold':100, 'knnDepth':7}
    shortTermMemory = []
    while iteration < 4:
        print('feedback:', feedback)
        prediction = graphInput(label, '/Users/kylejames/Desktop/teachAiToFeelSpring2022-main/happy1.jpeg',iteration,feedback)
        feedback = prediction.processFeedback()
        iteration+=1




#{"apiCoverageThreshold":90,"apiSuccessThreshold":90,"componentCoverageThreshold":90,"componentSuccessThreshold":90,"successRateApi":"","successRateComponent":"","testCoverageApi":"","testCoverageComponent":""}
{"id":"6090489f7ccf430318a7457e","scope":"workspace","date":"2022-04-11","aggregationType":"testCoverage"}
