import cv2
from lib import utils, model

YOLOv3 = {  "cfg":"cfg/yolov3.cfg", 
            "weights" :"weights/yolov3.weights"}

YOLOv3tiny = {  "cfg":"cfg/yolov3-tiny.cfg", 
                "weights":"weights/yolov3-tiny.weights"}

if __name__ == "__main__":
    img = cv2.imread("inputs/street.jpg")
    net = model.yolo(YOLOv3["cfg"], YOLOv3["weights"], "coco.names")
    boxes, confs = net.predict(img)
    Img = utils.draw_outputs(img, boxes, confs, show = False)
    utils.draw_distances(Img, boxes)
