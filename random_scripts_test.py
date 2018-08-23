import numpy as np
import cv2
import glob
from PIL import Image
import pandas as pd

# img = Image.open('16.jpg')
# image_array = np.asarray(img).astype(int).flatten()                         #.astype(np.float32)
# image_array2 = np.concatenate((np.array([1]),image_array),axis=0).astype(int)
# dataset = pd.read_csv('train.csv')
# dataset_list =['ans']
# for x in range(8101):
#     dataset_list.append('pixel'+str(x))


f=open('train.csv','ab')
# image_array2 = np.concatenate((np.array([1]),image_array),axis=0).astype(np.int16)
# np.savetxt(f, image_array2, delimiter="\\n", newline=',')
# f.write(b'\n')
# image_array2 = np.concatenate((np.array([1]),image_array),axis=0).astype(np.int16)
# np.savetxt(f, image_array2, delimiter="\\n", newline=',')
# f.write(b'\n')
i=0

for filename in glob.iglob('D:\\mydataset\\negative-fin\\*', recursive=True):
    img = Image.open(filename)
    image_array = np.asarray(img).astype(int).flatten()
    # image_array2 = np.concatenate((np.array([0]), image_array), axis=0).astype(int)
    image_array2 = np.concatenate((np.array([0]), image_array), axis=0).astype(np.int16)
    np.savetxt(f, image_array2, delimiter="\\n", newline=',')
    f.write(b'\n')
    i = i + 1
    print(i)


# image_array2 = np.reshape(image_array,(90,90))
# img2 = Image.fromarray(image_array2)
# img2 = img.resize((90, 90), Image.ANTIALIAS)
# img2.save('wow.png')
# img.save('wowg.png')
# print(image_array)
# print(image_array)



# img.save('16-grey.png')
# img = Image.open('17.jpg').convert('L')
# img = img.resize((90, 90), Image.ANTIALIAS)
# img.save('17-grey.png')


