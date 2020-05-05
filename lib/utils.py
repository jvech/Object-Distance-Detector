import cv2

def draw_outputs(img, boxes, confs, show = True):
    Img = img.copy()
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        Img = cv2.circle(Img, (x,y), 3, (0,0,255), -1)
        Img = cv2.putText(Img,"{:1.2f}".format(confs[i]), (x,y), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255)) 
    if show == True:
        while(1):
            cv2.imshow("Foto", Img)
            if cv2.waitKey(1) == ord('q'):
                break
    return Img
