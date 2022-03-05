from atoms import Atom 
from molecules import Molecule
from cells import Cell
from tissue import Tissue

label = 'predict'
imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/cas.jpg'

class graphInput():
    
    def __init__(self,label,imgPath):
        self.label = label 
        self.imgPath = imgPath
        # Process Face Data (DPR)
        atom = Atom(label, imgPath)
        atom.processEyes()
        atom.proccessImg()
        atom.createMolecule(label)
        # Process Emotion (KNN)
        molecule = Molecule(label, atom.moleculeImgPath)
        print('x:',  molecule.x)
        print('y:',  molecule.y)
        molecule.train()
        # Analogus Reasoning
        cell = Cell(molecule)
        cell.gatherAnalogiesView(molecule)
        cell.createAnalogies(molecule)




if __name__ == '__main__':
    prediction = graphInput(label, imgPath)
