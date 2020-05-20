import cv2
import numpy as np
from math import factorial
from numpy.linalg import norm

comb = lambda n: factorial(n)//(2*factorial(n-2))


def draw_outputs(img, boxes, confs, show = True):
    Img = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        # x, y, w, h = boxes[i,:]
        # Img = cv2.circle(Img, (x,y), 3, (0,0,255), -1)
        Img = cv2.rectangle(Img, (x,y), (x+w,y+h), (0,0,255), 2)
        Img = cv2.putText(Img,"{:1.2f}".format(confs[i]), (x,y), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    if show == True:
        while(1):
            cv2.imshow("Foto", Img)
            if cv2.waitKey(1) == ord('q'):
                break
    return Img

def draw_distances(img, Boxes, show = True):
    try:
        x, y, w, h = (Boxes.T)[:]
        x, y = x + w//2, y + h//2
        boxes = np.array([x,y,w,h]).T
        dist = distance(boxes)
        Img = img.copy()
        for i, X in enumerate(dist):
            d, x1, x2 = X
            Img = cv2.line(Img, tuple(x1.tolist()), tuple(x2.tolist()), (0,255,0), thickness=1)
        if show:
            while cv2.waitKey(1) != ord('q'):
                cv2.imshow("Foto", Img)
    except ValueError:
        Img = img.copy()
    finally:
        if show:
            while cv2.waitKey(1) != ord('q'):
                cv2.imshow("Foto", Img)
        return Img


def distance(boxes):
    if boxes.shape[0] <= 1:
        raise ValueError ("you must have 2 or more boxes")
    n = comb(boxes.shape[0])
    indxs = []
    d = []
    for i in range(boxes.shape[0]):
        x1 = boxes[i,0:2]
        for j in range(boxes.shape[0]):
            if j not in indxs and j != i:
                x2 = boxes[j,0:2]
                d.append([norm(x1-x2).astype(np.int32), x1, x2])
        indxs.append(i)
        if len(indxs) == n:
            break
    return np.array(d) 