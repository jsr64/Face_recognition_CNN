from darkflow.net.build import TFNet
import cv2
from PIL import Image
# #
# # options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/yolov2-tiny-voc.weights", "threshold": 0.2}
# #
# # tfnet = TFNet(options)
# #
# # imgcv = cv2.imread("test.jpg")
# # result = tfnet.return_predict(imgcv)
# # # print(type(result))
# # # print(type(result[0]))
# # print(result)
# # print(imgcv.shape[0])   #height widhth channel
# # print(result[0]["label"])
#
# cv2.imwrite('test-result.png',img)
# img = Image.open('test-result.png').convert('L')
# img = img.resize((90, 90))
# img.save('test-result.png')

def extract_result_set(file_name):
    options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/yolov2-tiny-voc.weights", "threshold": 0.5}
    tfnet = TFNet(options)
    imgcv = cv2.imread(file_name)
    result = tfnet.return_predict(imgcv)
    return result