import cv2
import dlib
import collections
import numpy as np

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
            shape = self.getEyes()

            for (x,y) in shape:
                cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
                # if (x, y) in self.leftEye:
                #     cv2.circle(self.img, (x, y), 2, (255, 0, 0), -1)
                # if (x, y) in self.rightEye:
                #     cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)

    def getEyes(self):

        leftEye,rightEye =[36,37,38,39,40,41],[42,43,44,45,46,47]
       
        # left  x1:36, x2:39, y1:41, y2:38
        # right x1:42, x2:45, y1:46, y2: 44
        lEye, rEye = [], []
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (self.shape.part(i).x, self.shape.part(i).y)
            # if i in leftEye:
            #     lEye.append((self.shape.part(i).x, self.shape.part(i).y))
            # if i in rightEye:
            #     rEye.append((self.shape.part(i).x, self.shape.part(i).y))
        
        #print(coords)
        self.leftEyeCords =  {'x1': coords[36][0], 'x2': coords[39][0], 'y1':coords[38][1], 'y2': coords[41][1]}
        self.rightEyeCords = {'x1': coords[42][0], 'x2': coords[45][0], 'y1':coords[43][1], 'y2': coords[46][1]}
        self.coords = coords
        # self.leftEye = lEye
        # self.rightEye = rEye
        return coords

    def processEyes(self):
        # cropped = img[start_row:end_row, start_col:end_col]
        img = self.img
        #crop_img = img[y:y+h, x:x+w]
        self.leftEyeImg = img[self.leftEyeCords['y1']-10: self.leftEyeCords['y2']+10, 
                                self.leftEyeCords['x1']-10:self.leftEyeCords['x2']+10].copy()
        
        self.rightEyeImg = img[self.rightEyeCords['y1']-10: self.rightEyeCords['y2']+10, 
                                self.rightEyeCords['x1']-10:self.rightEyeCords['x2']+10].copy()


         

if __name__ == "__main__":
    # face = dlibCords('/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/happy.png')
    # face.getFaceLandMarks()
    # cv2.imshow('image window', face.img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    atomOne = Atom('Joy','/Users/kjams/Desktop/dataAnalysis2022Spring/images/images/cas.jpg' )
    atomOne.proccessImg()
    atomOne.processEyes()
    cv2.imshow('bork', atomOne.leftEyeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    cv2.imshow('bork', atomOne.rightEyeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()