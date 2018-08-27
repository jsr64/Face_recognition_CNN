import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import to_categorical


# import tensorflow as tf
# tf.control_flow_ops = tf



dataset = pd.read_csv("data/train.csv").values

# X = dataset[dataset.columns[1:]].values
# Y = dataset[dataset.columns[0:1]].values
# x_train, y_train = shuffle(X, Y, random_state=89)

# trainX = dataset[:,1:].reshape(dataset.shape[0],1,90,90).astype( 'float32' )
trainX = dataset[:,1:].reshape(dataset.shape[0],1,90,90).astype( 'float32' )
print(trainX.shape)
# y_train = y_train[:,0]
y_train = dataset[:,0]
y_train = to_categorical(y_train)

trainX,y_train=shuffle(trainX, y_train, random_state=86)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(1,90,90), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1200, activation='relu'))
model.add(Dense(2, activation='softmax'))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01),
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(trainX, y_train,
           batch_size=200,
          epochs=100,
          verbose=1
          )

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



