# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# Takes best three results from layertester run (manually inputted).
# Tests each layer/node count combination with between 5 and 20 epochs.
# Best result will be used in the main script.

# Import necessary libraries.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import loaddata

# Load data.
[train_images, train_labels, test_images, test_labels] = loaddata.ld()

# Setup node and layer count values.
layernode = [128, 128, 256]
layercount = [7, 9, 12]
# Preallocate list of accuracy results.
accuracylist = {}

# Setup and run model. First loop iterates through node/layer count values.
for i in range(0, len(layernode)):
  # Second loop iterates through epoch counts.
  for j in range(5,21):
    # Notify user of current iteration.
    print("Nodecount: " + str(layernode[i]) + " Layercount: " + str(layercount[i]))
    # Use sequential model.
    model = keras.Sequential()
    # Flatten image to 1D array.
    model.add(keras.layers.Flatten(input_shape=(200, 300)))
    # Add the number of dense layers as determined by layercount array.
    for k in range(0, layercount[i]):a
        # Each dense layer should have a nodecount determined by layernode array.
        model.add(keras.layers.Dense(layernode[i], activation=tf.nn.relu))
    # Add final layer for results.
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    # Compile model.
    model.compile(optimizer='Adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model with number of epochs determined by current iteration.
    model.fit(train_images, train_labels, epochs=j)

    # Test model.
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # Print and save accuracy.
    print('Test accuracy:', test_acc)
    accuracylist['N' + str(layernode[i]) + 'L' + str(j)] = test_acc

print(accuracylist)