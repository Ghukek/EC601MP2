# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# Loads datasets and makes them more uniform.

# Import necessary libraries.
import numpy as np

def ld():
    # Load datasets.
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")

    # Switch to floating point values.
    train_images = train_images.astype(float)
    test_images = test_images.astype(float)

    # Bring training data to a more uniform state.
    for i in range(0, len(train_images)):
      # Bring smallest value to zero.
      train_images[i] = train_images[i] - np.amin(train_images[i])
      # Bring largest value to one.
      train_images[i] = train_images[i]/np.amax(train_images[i])
      # Invert if average value is higher.
      if np.average(train_images[i]) > .5:
        train_images[i] = 1 - train_images[i]

    # Repeat for test data.
    for i in range(0, len(test_images)):
      test_images[i] = test_images[i] - np.amin(test_images[i])
      test_images[i] = test_images[i]/np.amax(test_images[i])
      if np.average(test_images[i]) > .5:
        test_images[i] = 1 - test_images[i]

    return [train_images, train_labels, test_images, test_labels]