# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

import tensorflow as tf
from tensorflow import keras
import numpy as np

train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

train_images = train_images.astype(float)
test_images = test_images.astype(float)

for i in range(0, len(train_images)):
  train_images[i] = train_images[i] - np.amin(train_images[i])
  train_images[i] = train_images[i]/np.amax(train_images[i])
  if np.average(train_images[i]) > .5:
    train_images[i] = 1 - train_images[i]

for i in range(0, len(test_images)):
  test_images[i] = test_images[i] - np.amin(test_images[i])
  test_images[i] = test_images[i]/np.amax(test_images[i])
  if np.average(test_images[i]) > .5:
    test_images[i] = 1 - test_images[i]

layernode = [32, 64, 128, 256, 512, 1024]
accuracylist = {}

for i in range(0, len(layernode)):
    for j in range(5, 16):
        print("Nodecount: " + str(layernode[i]) + " Layercount: " + str(j))
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(200, 300)))
        for k in range(0, j):
            model.add(keras.layers.Dense(layernode[i], activation=tf.nn.relu))
        model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

        model.compile(optimizer='Adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, epochs=5)

        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print('Test accuracy:', test_acc)

        accuracylist['N' + str(layernode[i]) + 'L' + str(j)] = test_acc

print(accuracylist)
