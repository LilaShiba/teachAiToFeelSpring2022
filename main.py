from atoms import Atom 
from molecule import Molecule

label = 'predict'
imgPath = '/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png'


class graphInput():
    
    def __init__(self,label,imgPath):
        self.label = label 
        self.imgPath = imgPath
        # Process Face Data
        atom = Atom(label, imgPath)
        atom.processEyes()
        atom.proccessImg()
        atom.createMolecule(label)
        # Process Emotion
        molecule = Molecule(label, atom.moleculeImgPath)
        print('x:',  molecule.x)
        print('y',  molecule.y)




if __name__ == '__main__':
    prediction = graphInput(label, imgPath)
