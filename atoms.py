import os 
import cv2
import dlib
import collections
import numpy as np



class Atom:

    def __init__(self,label,imgPath,iteration,faceOverlap):

        #empty bois
        self.x = None
        self.y = None 
        self.leftEyeCords = None 
        self.rightEyeCords = None
        self.leftEyeImg = [] 
        self.rightEyeImg = []
        self.leftEyeArray = []
        self.rightEyeArray = []
        self.cords = ()
        #lil process
        self.iteration = iteration
        self.threshold = faceOverlap
        self.label = label
        self.imgPath = imgPath 
        self.img = cv2.imread(imgPath)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        self.rects = detector(self.gray,1)
        # creates, self.coords,self.leftEye,self.rightEye,& self.shape
        self.proccessImg()

    def proccessImg(self):
        predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        for (i, rect) in enumerate(self.rects):
          
            self.shape = predictor(self.gray, rect)
            shape = self.getEyes()

    def getEyes(self):
        leftEye  = [36,37,38,39,40,41]
        rightEye = [42,43,44,45,46,47]
        if self.shape:
            lEye, rEye = [], []
            coords = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                coords[i] = (self.shape.part(i).x, self.shape.part(i).y)
              
            self.leftEyeCords =  {'x1': coords[36][0], 'x2': coords[39][0], 'y1':coords[38][1], 'y2': coords[41][1]}
            self.rightEyeCords = {'x1': coords[42][0], 'x2': coords[45][0], 'y1':coords[43][1], 'y2': coords[46][1]}
        
            self.coords = coords
            self.processEyes()
            
    def processEyes(self):
        # cropped = img[start_row:end_row, start_col:end_col]
        k = self.threshold
        img = self.img
        #crop_img = img[y:y+h, x:x+w]
        # We only want data with two eyes for basic training
        if self.leftEyeCords and self.rightEyeCords:
            self.leftEyeImg = img[self.leftEyeCords['y1']-k: self.leftEyeCords['y2']+k, 
                                    self.leftEyeCords['x1']-k:self.leftEyeCords['x2']+k].copy()
            
            self.rightEyeImg = img[self.rightEyeCords['y1']-k: self.rightEyeCords['y2']+k, 
                                    self.rightEyeCords['x1']-k:self.rightEyeCords['x2']+k].copy()

            self.leftEyeArray  =  np.array(self.leftEyeImg.copy())
            self.rightEyeArray =  np.array(self.rightEyeImg.copy())
         
    def createMolecule(self,title):
        #if (atomOne.rightEyeArray) > 0 and len(atomOne.rightEyeArray) > 0:
        count = 0
        # if both eyes found
        delta = 'prediction_'+str(self.iteration)
        if len(self.rightEyeImg) > 1 and len(self.leftEyeImg) > 1:
           
            os.makedirs(delta)
            cv2.imwrite(delta+'/LeftEye.png', self.leftEyeImg)
            cv2.imwrite(delta+'/RightEye.png', self.rightEyeImg)
            self.moleculeImgPath = delta
            count+=1
        else:
            print('picture quality does not suffice')
        
        self.delta = delta