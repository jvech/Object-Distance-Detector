import cv2
from lib import utils, model

YOLOv3 = {  "cfg":"cfg/yolov3.cfg", 
            "weights" :"weights/yolov3.weights"}

YOLOv3tiny = {  "cfg":"cfg/yolov3-tiny.cfg", 
                "weights":"weights/yolov3-tiny.weights"}

if __name__ == "__main__":
    net = model.yolo(YOLOv3tiny["cfg"], YOLOv3tiny["weights"], "coco.names")
    video = cv2.VideoCapture("inputs/Cargando_cemento.mp4")
    ret = True
    while video.isOpened():
        ret, frame = video.read()
        boxes, confs = net.predict(frame)
        frame = utils.draw_outputs(frame, boxes, confs, show=False)
        frame = utils.draw_distances(frame, boxes, show=False)
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            video.release()
            break
    cv2.destroyAllWindows()
