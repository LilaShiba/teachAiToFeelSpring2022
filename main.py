import os
import cv2 
from PIL import Image
from cells import Cell
from atoms import Atom 
from tissue import Tissue
from molecules import Molecule
import matplotlib.pyplot as plt


class graphInput():
    
    def __init__(self,label,imgPath,iteration):
        self.label = label 
        self.imgPath = imgPath
        self.iteration =  iteration
        # Process Face Data (DPR)
        # on init,  will run
        atom = Atom(label, imgPath, faceOverlap=4) 
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
        return tissue.processFeedback()
        



if __name__ == '__main__':
    label = 'predict'
    iteration = 0
    feedBack = graphInput(label, '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/39843011-angry-face-man.webp',iteration)


    # while iteration < 3:
    #     prediction = graphInput(label, '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/39843011-angry-face-man.webp',iteration)
    #     iteration+=1
