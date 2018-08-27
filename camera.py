from darkflow.net.build import TFNet
import cv2
from PIL import Image

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/yolov2-tiny-voc.weights", "threshold": 0.2}

tfnet = TFNet(options)

imgcv = cv2.imread("1.jpg")
result = tfnet.return_predict(imgcv)
# print(type(result))
# print(type(result[0]))
print(result)
print(imgcv.shape[0])   #height widhth channel
print(result[0]["label"])
for person in result:
    topleft_x = person['topleft']['x']
    topleft_y = person['topleft']['y']
    bottomright_x = person['bottomright']['x']
    bottomright_y = person['bottomright']['y']
    if((topleft_y+imgcv.shape[0]*0.08)>imgcv.shape[0]):
        topleft_y = int(imgcv.shape[0])
    else:
        topleft_y = int(topleft_y - (imgcv.shape[0] * 0.1))
    bottomright_y = int(bottomright_y - (imgcv.shape[0] * 0.1))
    img = cv2.rectangle(imgcv, (topleft_x, topleft_y), (bottomright_x, bottomright_y), (0, 255, 0), 5)

cv2.imwrite('test-result.png',img)


img = Image.open('test-result.png').convert('L')
img = img.resize((90, 90))
img.save('test-result.png')

