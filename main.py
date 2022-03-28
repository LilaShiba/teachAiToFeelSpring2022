from atoms import Atom 
from molecules import Molecule
from cells import Cell
from tissue import Tissue
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import os



#imgPath ='/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/angryTest.jpeg'
#imgPath = 'images/validation/fear/7748.jpg'
#imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/validation/sad/98.jpg'
#imgPath = 'images/validation/disgust/807.jpg'
#imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/validation/happy/8.jpg'
#imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/validation/sad/798.jpg'
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
        #print(molecule.knnMap)
        # Analogus Reasoning
        cell = Cell(molecule.x, molecule.y, molecule.mapOfEmotions)
        cell.knn(7)
        # distro of feelings for working memory, e.g., result of cell.knn aggregated
        cell.gatherAnalogiesView()
        # cell.gatherAnalogiesView(molecule)
        # cell.createAnalogies(molecule)
        # Systems Brah aka tissue




if __name__ == '__main__':
    label = 'predict'
    imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png'
    prediction = graphInput(label, imgPath)