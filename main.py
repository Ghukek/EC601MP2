# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#print(tf.__version__)

# Import dataset.

train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

train_images = train_images.astype(float)
test_images = test_images.astype(float)

# Create list of labels.

class_names = ['Ship', 'Airplane']

# Bring values to a range of 0-1.

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

# Setup the layers

model = keras.Sequential()
# This layer turns the image from a 2D array to a 1D array.
model.add(keras.layers.Flatten(input_shape=(200, 300)))

for i in range(0,12):
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))

# This layer will return an array of 2 probability scores summing to 1.
# Each node's score reflects the probability that the image belongs to the
# corresponding class.
model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

# Compile with settings.

model.compile(optimizer='Adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Send training and test data to the model.
# Model will train using the training data.

model.fit(train_images, train_labels, epochs=15)

# Run test data through the model and print accuracy.

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Make predictions on images.

predictions = model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# Setup plot functions to plot probability that an image falls in each category.

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot a random set of 15 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  j = randint(0, 100)
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(j, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(j, predictions, test_labels)

plt.show()

# Some code in this script is taken from 
# https://www.tensorflow.org/tutorials/keras/basic_classification
# The code is under the Creative Commons Attribution 3.0 License.
# To see the original code, see included example01.py.

# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.