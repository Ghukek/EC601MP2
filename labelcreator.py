# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

import os
import imgnum
from PIL import Image
from numpy import*

# innum should be the number of images in the folder.
# filepath should the the path of the folder.

def createarrays(innum, filepath):
    # Create empty arrays for images and labels.
    images = []
    labels = []
    # Note folder we are checking for user to understand.
    print("Please check " + filepath + " for the following inputs: ")
    # Loop through images by number
    for i in range(1, innum+1):
        # Get file string according to standard number format.
        imgstr = imgnum.filestr(i)
        # Create variable temp.
        temp = []
        # Try to open image and save as temp. Notify and skip if failed.
        try:
            temp = asarray(Image.open(filepath + imgstr))
        except:
            print("Couldn't open: " + imgstr + " skipping.")
            pass
        # Ask if image is of an airplane.
        labelval = int(input("Is " + imgstr + " an airplane? Input 1 or 0: "))
        # Ensure only 0 or 1 are given.
        while True:
            if labelval == 0 or labelval == 1:
                break
            # Used for testing without going through entire list of images.
            elif labelval == 3:
                return (images, labels)
            else:
                print("Error: input not 0 or 1. Please input 0 for no or 1 for yes.")
                labelval = int(input("Is " + imgstr + " an airplane? Input 1 or 0: "))
        # Notify user of their choice.
        if labelval == 1:
            print("You selected airplane.")
        else:
            print("You selected ship.")
        # Add image and label to corresponding list.
        images.append(temp)
        labels.append(labelval)

    # Convert list to array to fit with appropriate format.
    images = array(images)
    labels = array(images)

    return (images, labels)

# First figure out how many files are in the train data.
trainnum = len(os.listdir(traindir))
# Send to the array creator.
(train_images, train_labels) = createarrays(trainnum, "./traindata/")
# Repeat for test data.
testnum = len(os.listdir(testdir))
(test_images, test_labels) = createarrays(testnum, "./testdata/")

# Show size of resulting arrays.
print("Train Images Size: " + str(train_images.shape))
print("Train Labels Size: " + str(train_labels.shape))
print("Test Images Size: " + str(test_images.shape))
print("Test Labels Size: " + str(test_labels.shape))

# Save arrays to file.
save("train_images", train_images)
save("train_labels", train_labels)
save("test_images", test_images)
save("test_labels", test_labels)
