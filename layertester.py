# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# Tests different layer/node count combinations.
# Best result will be further tested for ideal epoch count.

# Import necessary libraries.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import loaddata

# Load data.
[train_images, train_labels, test_images, test_labels] = loaddata.ld()

# Setup nodecount values.
layernode = [32, 64, 128, 256, 512, 1024]
# Preallocate accuracy list.
accuracylist = {}

# Setup and run model. First loop iterates through nodecount values.
for i in range(0, len(layernode)):
  # Second loop iterates through layer count values.
  for j in range(5, 16):
    # Notify user of current iteration.
    print("Nodecount: " + str(layernode[i]) + " Layercount: " + str(j))
    # Use sequential model.
    model = keras.Sequential()
    # Flatten image to 1D array.
    model.add(keras.layers.Flatten(input_shape=(200, 300)))
    # Add the number of dense layers as determined by current iteration.
    for k in range(0, j):
      # Each dense layer should have a nodecount determined by layernode array.
      model.add(keras.layers.Dense(layernode[i], activation=tf.nn.relu))
    # Add final layer for results.
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    # Compile model.
    model.compile(optimizer='Adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model.
    model.fit(train_images, train_labels, epochs=5)

    # Test model.
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Print and save test accuracy.
    print('Test accuracy:', test_acc)
    accuracylist['N' + str(layernode[i]) + 'L' + str(j)] = test_acc

print(accuracylist)
