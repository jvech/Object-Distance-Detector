import cv2
from lib import utils, model

YOLOv3 = {  "cfg":"cfg/yolov3.cfg", 
            "weights" :"weights/yolov3.weights"}

YOLOv3tiny = {  "cfg":"cfg/yolov3-tiny.cfg", 
                "weights":"weights/yolov3-tiny.weights"}

if __name__ == "__main__":
    img = cv2.imread("inputs/people.jpeg")
    net = model.yolo(YOLOv3tiny["cfg"], YOLOv3tiny["weights"], "coco.names")
    boxes, confs = net.predict(img)
    utils.draw_outputs(img, boxes, confs)