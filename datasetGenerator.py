import numpy as np
import cv2
import glob
i = 0

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



for filename in glob.iglob('D:\lfw\lfw\*\*', recursive=True):
    img = cv2.imread(filename)
    rimg = img.copy()
    rimg1 = rotateImage(rimg, 15)
    cv2.imwrite("D:\\neg1\\"+str(i)+".jpg", img);
    i = i+1
    rimg2 = rotateImage(rimg, -15)
    cv2.imwrite("D:\\pos1\\" + str(i) + ".jpg", rimg2);
    i = i + 1



img=cv2.imread('oo.jpg')
rimg=img.copy()
rimg=cv2.flip(img,1)
cv2.imwrite("D:\\pos\\ray_Image.jpg", rimg);
abc = rotateImage(img,-10)
cv2.imwrite("ray_Image_rotat.jpg", abc);