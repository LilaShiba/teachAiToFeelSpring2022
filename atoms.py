import cv2
import dlib
import collections
import numpy as np


class dlibCords():

    def __init__(self, imgPath):
        self.img = cv2.imread(imgPath)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.detector = dlib.get_frontal_face_detector()
        self.rects = self.detector(self.gray,1)

    def shape_to_np(self, shape):
        leftEye=[36,37,38,39,40,41]
        rightEye=[42,43,44,45,46,47]

        lEye = []
        rEye = []

        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (self.shape.part(i).x, self.shape.part(i).y)
            if i in leftEye:
                lEye.append((self.shape.part(i).x, self.shape.part(i).y))
            if i in rightEye:
                rEye.append((self.shape.part(i).x, self.shape.part(i).y))
        #print(coords)
        self.coords = coords
        self.leftEye = lEye
        self.rightEye = rEye
        return coords

    def getFaceLandMarks(self):
        predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        for (i, rect) in enumerate(self.rects):
            self.shape = predictor(self.gray, rect)
            self.shape = self.shape_to_np(self.shape)
            for (x,y) in self.shape:
                cv2.circle(self.img, (x, y), 2, (0, 255, 0,0), -1)

            for (x, y) in self.leftEye:
                cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
            for (x, y) in self.rightEye:
                cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)
        
  




class Atom:

    def __init__(self,label,imgPath):
        self.x = None
        self.y = None 
        self.label = label
        self.cords = ()
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
            self.shape = self.shape_to_np()
            for (x,y) in self.shape:
                cv2.circle(self.img, (x, y), 2, (0, 255, 0,0), -1)
                if (x, y) in self.leftEye:
                    cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
                if (x, y) in self.rightEye:
                    cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)

    def shape_to_np(self):

        leftEye,rightEye =[36,37,38,39,40,41],[42,43,44,45,46,47]
        lEye, rEye = [], []
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (self.shape.part(i).x, self.shape.part(i).y)
            if i in leftEye:
                lEye.append((self.shape.part(i).x, self.shape.part(i).y))
            if i in rightEye:
                rEye.append((self.shape.part(i).x, self.shape.part(i).y))
        #print(coords)
        self.coords = coords
        self.leftEye = lEye
        self.rightEye = rEye
        return coords



if __name__ == "__main__":
    # face = dlibCords('/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png')
    # face.getFaceLandMarks()
    # cv2.imshow('image window', face.img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    atomOne = Atom('Joy','/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png' )
    atomOne.proccessImg()
    cv2.imshow('image window', atomOne.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()