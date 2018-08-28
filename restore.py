import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import model_from_json
import camera
import cv2
from PIL import Image
import numpy as np

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
dataset = pd.read_csv("data/train.csv").values

# X = dataset[dataset.columns[1:]].values
# Y = dataset[dataset.columns[0:1]].values
# x_train, y_train = shuffle(X, Y, random_state=89)

trainX = dataset[2,1:].reshape(1,1,90,90).astype( 'float32' )
# y_train = y_train[:,0]
y_train = dataset[:,0]
y_train = to_categorical(y_train)
score = loaded_model.predict(trainX)
print(y_train.shape)
print(y_train[5,:])
# print(score)

i=0
result = camera.extract_result_set("test.jpg")
imgcv = cv2.imread("test.jpg")
print(result)
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
    # crop_img = img[topleft_y:bottomright_y, topleft_x:bottomright_x]
    crop_img=img
    cv2.imwrite('temp'+str(i)+'.png', crop_img)
    img = Image.open('temp'+str(i)+'.png').convert('L')
    i=i+1
    img = np.array(img.resize((90, 90))).reshape(1,1,90,90).astype( 'float32' )
    a = loaded_model.predict(img)
    print(a)
