from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/yolov2-tiny-voc.weights", "threshold": 0.1}

tfnet = TFNet(options)
imgcv = cv2.imread("oo.jpg")
result = tfnet.return_predict(imgcv)
print(result)