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