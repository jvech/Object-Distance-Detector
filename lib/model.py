import cv2
import numpy as np

def load_image(img_path):
    img = cv2.imread(img_path)/255
    img = cv2.resize(X, None, fx=0.4, fy=0.4)
    height, width, ch = img.shape
    return img, height, width, ch

class yolo():
    def __init__(self, cfg_file, weight_file, class_file):
        with open(class_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.model = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)
        layers_names = self.model.getLayerNames()
        self.out_layers = [layers_names[i[0]-1] for i in self.model.getUnconnectedOutLayers()]

        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
    

    def predict(self, img):
        height, width, ch = img.shape
        X = cv2.resize(img, None, fx=0.4, fy=0.4)
        blob = cv2.dnn.blobFromImage(X, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward(self.out_layers)
        ## get boxes
        boxes = []
        confs = []
        for out in outputs:
            for detect in out:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3 and self.classes[class_id] == "person":
                    x = int(detect[0] * width)
                    y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    boxes.append([x, y, w, h])
                    confs.append(conf)
        return np.array(boxes), confs
    
    def predict_video(self, blob, shape):
        ch, height, width = shape
        self.model.setInput(blob)
        outputs = self.model.forward(self.out_layers)
        ## get boxes
        boxes = []
        confs = []
        for out in outputs:
            for i in range(blob.shape[0]):
                for detect in out[i]:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > 0.3 and self.classes[class_id] == "person":
                        x = int(detect[0] * width)
                        y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        boxes.append([i, x, y, w, h])
                        confs.append([i, conf])
        return np.array(boxes), np.array(confs)