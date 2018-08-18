import numpy as np
import cv2
import glob
from PIL import Image
i = 0

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


#
# for filename in glob.iglob('D:\mydataset\positive\*', recursive=True):
#     img = cv2.imread(filename)
#     cv2.imwrite("D:\\mydataset\\positive-fin-noreshape\\" + str(i) + ".jpg", img);
#     i = i + 1
#     print(i)
#     rimg = rotateImage(img, 18)
#     cv2.imwrite("D:\\mydataset\\positive-fin-noreshape\\" + str(i) + ".jpg", rimg);
#     i = i + 1
#     rimg2 = rotateImage(img, -18)
#     print(i)
#     cv2.imwrite("D:\\mydataset\\positive-fin-noreshape\\" + str(i) + ".jpg", rimg2);
#     i = i + 1
#     print(i)

#
# for filename in glob.iglob('D:\\mydataset\\negative\\*', recursive=True):
#     img = cv2.imread(filename)
#     if(i%4==0):
#         rimg2 = rotateImage(img, -18)
#         cv2.imwrite("D:\\mydataset\\negative-fin\\" + str(i) + ".jpg", rimg2)
#     elif(i%6==0):
#         rimg = rotateImage(img, 18)
#         cv2.imwrite("D:\\mydataset\\negative-fin\\" + str(i) + ".jpg", rimg)
#     else:
#         cv2.imwrite("D:\\mydataset\\negative-fin\\" + str(i) + ".jpg", img)
#     print(i)
#     i = i + 1

#
#
# img=cv2.imread('oo.jpg')
# rimg=img.copy()
# rimg=cv2.flip(img,1)
# cv2.imwrite("D:\\pos\\ray_Image.jpg", rimg);
# abc = rotateImage(img,-10)
# cv2.imwrite("ray_Image_rotat.jpg", abc);
#
# for filename in glob.iglob('D:\mydataset\positive\*', recursive=True):
#     img = Image.open(filename).convert('L')
#     img = img.resize((90, 90))
#     img.save("D:\\mydataset\\positive-fin\\" + str(i) + ".jpg");
#     i = i + 1
#     print(i)


for filename in glob.iglob('D:\\mydataset\\negative\\*', recursive=True):
    img = Image.open(filename).convert('L')
    img = img.resize((90, 90))
    img.save("D:\\mydataset\\negative-fin\\" + str(i) + ".jpg")
    i = i + 1
    print(i)