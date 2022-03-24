from atoms import Atom 
from molecules import Molecule
from cells import Cell
from tissue import Tissue
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import os


label = 'predict'
#imgPath ='/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/validation/angry/966.jpg'
#imgPath = 'images/validation/fear/7748.jpg'
imgPath = '/Users/kylejames/Desktop/robitFeelings/teachAiToFeelSpring2022/testing/self.jpg'
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
        cell = Cell(molecule.x, molecule.y, molecule.knnMap)
        cell.knn(5)
        # collection of edges
        for array in cell.edges['molecule']:
            print('array:')
            print(array)
            feeling,molecule = array[0][0],array[0][1]
            #folderNumber = molecule[1].filePath.split("/")

            files = os.listdir(molecule.filePath)
            deltaPathL = molecule.filePath +'/'+files[0] 
            deltaPathR = molecule.filePath +'/'+files[1] 
            print('edge left eye:')
            print('deltaPathL:', deltaPathL)
            l = Image.open(deltaPathL)
            l.show()
            

            print('edge right eye:')
            print('deltaPathR:', deltaPathR)
            r = Image.open(deltaPathR)
            r.show()
            break  
        # cell.gatherAnalogiesView(molecule)
        # cell.createAnalogies(molecule)
        # Systems Brah aka tissue




if __name__ == '__main__':
    prediction = graphInput(label, imgPath)
