import os
import cv2 
from PIL import Image
from cells import Cell
from atoms import Atom 
from tissue import Tissue
from molecules import Molecule
import matplotlib.pyplot as plt


class graphInput():
    
    def __init__(self,label,imgPath):
        self.label = label 
        self.imgPath = imgPath
        # Process Face Data (DPR)
        # on init,  will run
        atom = Atom(label, imgPath) 
        # atom.proccessImg() 
        # atom.processEyes()  
        # create prediction folder / moleculeImgPath
        atom.createMolecule(label)
        # Process Emotion (KNN)
        # On init, getDpr runs
        molecule = Molecule(label, atom.moleculeImgPath)
        print('x:',  molecule.x)
        print('y:',  molecule.y)
        # knn graph where 
        # xAxis=dprRightEye, yAxis=dprLeftEye , hue=label
        molecule.train()
        #molecule.showMap()
        #print(molecule.knnMap)
        # Analogus Reasoning
        cellNetwork = Cell(molecule.x, molecule.y, molecule.mapOfEmotions)
        cellNetwork.knn(10)
        # distro of feelings for working memory, e.g., result of cell.knn aggregated 
        cellNetwork.gatherAnalogiesView()
        # cell.createAnalogies(molecule)
        # Systems Brah aka tissue
        tissue = Tissue(cellNetwork)
        tissue.feedback()




if __name__ == '__main__':
    label = 'predict'
    #imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png'
    #imgPath = 'testing/self.jpg'
    prediction = graphInput(label, '/testing/self.jpg')
