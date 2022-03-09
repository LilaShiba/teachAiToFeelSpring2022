from atoms import Atom 
from molecules import Molecule
from cells import Cell
from tissue import Tissue

label = 'predict'
#imgPath ='/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/validation/angry/966.jpg'
imgPath = 'images/validation/fear/7748.jpg'
class graphInput():
    
    def __init__(self,label,imgPath):
        self.label = label 
        self.imgPath = imgPath
        # Process Face Data (DPR)
        # on init, atom.proccessImg(), atom.processEyes() will run
        atom = Atom(label, imgPath)   
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
        # Analogus Reasoning
        cell = Cell(molecule)
        cell.gatherAnalogiesView(molecule)
        cell.createAnalogies(molecule)
        # Systems Brah aka tissue




if __name__ == '__main__':
    prediction = graphInput(label, imgPath)
