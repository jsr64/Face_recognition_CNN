import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import model_from_json


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
print(y_train[2,:])
print(score)
