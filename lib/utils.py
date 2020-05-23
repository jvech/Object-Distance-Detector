import cv2
import numpy as np
from math import factorial
from numpy.linalg import norm

comb = lambda n: factorial(n)//(2*factorial(n-2))


def draw_outputs(img, boxes, confs, show = True, color = (0,0,255)):
    """
    Dibuja las bounding boxes junto con su respectiva fiabilidad
        img : imagen
        boxes : np.array([[x1,y1,w1,h1],...,[xn, yn, wn, hn]])
        confs : np.array([conf1, ..., confn])
        show  : True -> mostrar imagen con bounding boxes
                False -> no mostrar nada
        color -> Bounding Boxes Color (B,G,R)
    """
    Img = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        Img = cv2.rectangle(Img, (x,y), (x+w,y+h), color, 2)
        Img = cv2.putText(Img,"{:1.2f}".format(confs[i]), (x,y), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color)
    if show == True:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return Img

def draw_distances(img, Boxes, show = True):
    """
    Dibuja las distancias calculadas entre bounding boxes
        img : imagen
        boxes : np.array([[x1,y1,w1,h1],...,[xn, yn, wn, hn]]) shape = (n,4)
        confs : np.array([conf1, ..., confn]) shape = n
        show  : True -> mostrar imagen con bounding boxes
                False -> no mostrar nada
    """
    try:
        x, y, w, h = (Boxes.T)[:]
        x, y = x + w//2, y + h//2
        boxes = np.array([x,y,w,h]).T
        dist = distance(boxes)
        Img = img.copy()
        for i, X in enumerate(dist):
            d, x1, x2 = X
            Img = cv2.line(Img, tuple(x1.tolist()), tuple(x2.tolist()), (0,255,0), thickness=1)
    except ValueError:
        Img = img.copy()
    finally:
        if show:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Image', Img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return Img

def nearest(in_box, boxes):
    """
    me retorna el bounding box mas cercano a in_box
        in_box : np.array([x,y,w,h]) shape = (1,4)
        boxes  : np.array([[x,y,w,h]...]) shape = (n,4)
    """
    x1 = in_box[0:2]
    d, inds = [], []
    for i, x2 in enumerate(boxes[:,0:2]):
        if IoU(boxes[i][0:4],in_box) > 0.01:
            d.append(np.linalg.norm(x1-x2))
            inds.append(i)
    if len(d) != 0:
        return boxes[inds[np.argmin(d)]]
    else:
        return np.array([0, 0, 0, 0])

def Average_Prec(boxes_det, boxesB, confs):
    """
    Calcula el Average Precision
        boxes_det \t : Bounding boxes detectadas
        boxesB  \t : Bounding boxes etiquetadas
        confs \t : nivel de fiabilidad
    """
    n = len(boxes_det)
    Ap = 0
    indxs = np.argsort(confs)[::-1]
    boxesA = boxes_det[np.argsort(confs)[::-1]]
    for i, boxA in enumerate(boxesA):
        boxB = nearest(boxA, boxesB)
        if IoU(boxA, boxB) > 0.5:
            Ap += (Ap+1)/(i+1)
        else:
            Ap += Ap/(i+1)
    return Ap/n



def distance(boxes):
    """
    Calcula la distancia entre todos los bounding boxes
        boxes\t:np.array([[x,y,w,h],...]) shape = (n,4)
    """
    if len(boxes) <= 1:
        raise ValueError ("you must have 2 or more boxes")
    n = comb(len(boxes))
    indxs = []
    d = []
    for i in range(len(boxes)):
        x1 = boxes[i,0:2]
        for j in range(len(boxes)):
            if j not in indxs and j != i:
                x2 = boxes[j,0:2]
                d.append([norm(x1-x2).astype(np.int32), x1, x2])
        indxs.append(i)
        if len(indxs) == n:
            break
    return np.array(d)

def IoU(BoxA, BoxB):
    """
    Intersection over Union
    """
    assert BoxA.size == 4 & BoxB.size == 4, ("Incorrect Box Shape")
    I_x, I_y = np.max([BoxA[0:2], BoxB[0:2]], axis=0)
    I_w, I_h = np.min([BoxA[0:2] + BoxA[2:4], BoxB[0:2] + BoxB[2:4]], axis=0)

    I_A = max(0, I_w - I_x + 1) * max(0, I_h - I_y + 1)
    U_A = BoxA[2]*BoxA[3] + BoxB[2]*BoxB[3] - I_A
    return I_A/U_A

def NMS(boxes,confs, nms_thresh=0.4):
    """
    Calculates Non-Maximum Supression
        boxes \t: np.array([[x,y,w,h],...]) shape = (n,4)
        confs \t: np.array([conf1,...,]) shape = n
        nms_thres\t: IoU threshold
    """
    assert type(boxes) == np.ndarray, ("boxes must be np.ndarray")
    nms_boxes = []
    nms_confs = []
    bboxes = boxes.copy().tolist()
    cconfs = confs.copy().tolist()
    while bboxes != []:
        i = np.argmax(cconfs)
        M = bboxes.pop(i)
        nms_boxes.append(M)
        nms_confs.append(cconfs.pop(i))
        for j,b in enumerate(bboxes):
            if IoU(np.array(M),np.array(b)) >= nms_thresh:
                bboxes.pop(j)
                cconfs.pop(j)
    return np.array(nms_boxes), np.array(nms_confs)


