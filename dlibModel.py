import cv2
import dlib
import numpy as np

img = cv2.imread('/Users/kjams/Desktop/dataAnalysis2022Spring/images/happy.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
gray = np.asarray(grey)
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1)


print('rects:', rects)
print('')



def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat.bz2', [x,y,w,h])
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    cv2.imshow('bork',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
