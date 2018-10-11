# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# Takes from a selection of suitable photos.
# Assigns them to either test or train data.
# Converts to black and white.
# Resizes to 300x200.

import os
from PIL import Image
import imgnum
import random

# Define input directory.

fromdir = "./ImagesW1K"

# Define output directories.
# ..Indicates that we will be referencing them from the fromdir.

testdir = "../testdata/"
traindir = "../traindata/"

# Initialize filecounters for both datasets.

testnum = 0
trainnum = 0

# Change to input directory.

os.chdir(fromdir)
print("Pulling from: " + os.getcwd())

for filename in os.listdir():
    print("Getting: " + filename)
    # First decide whether the image will be training or testdata.
    if random.uniform(0, 1) > .8:
        outdir = testdir
        testnum = testnum + 1
        outnum = testnum
        print("Will be test picture #" + str(testnum))
    else:
        outdir = traindir
        trainnum = trainnum + 1
        outnum = testnum
        print("Will be train picture #" + str(trainnum))

    # Open image.
    try:
        curimg = Image.open(filename)
    except:
        pass
    print("Opened: " + filename)
    # Convert to black and white.
    curimg = curimg.convert('L')
    # Resize image to 300x200 pixels.
    curimg = curimg.resize((300, 200))
    # Generate savefile name.
    imgstr = imgnum.filestr(outnum)
    # Save into EC602MP2 directory.
    print(outdir + imgstr)
    curimg.save(outdir + imgstr)

print("Number of test images: " + str(testnum))
print("Number of train images: " + str(trainnum))

