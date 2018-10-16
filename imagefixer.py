# Copyright 2018 Nathan Wiebe nwiebe@bu.edu

# This script creates a directory full of sample images in a standard size.
# It draws from my personal collection of images of model shis and planes.
# This script does not need to be rerun. It has created images in ./ImagesW1K
# All images Copyright 2018 Nathan Wiebe dba Ghukek Miniatures ghukek@gmail.com

import os
from PIL import Image
import imgnum

# Create an array of directories to call from.

alldir = "/home/nathan/Pictures"
dirstr = ("/BlenderRenders", "/Miniatures", "/Miniatures/Older")
todir = "/home/nathan/Documents/EC601/Project2/EC601MP2/ImagesW1K/"

inum = 0

# Loop through chosen directories.
for directory in dirstr:
    # Change to directory.
    os.chdir(alldir+directory)
    print("Checking: " + os.getcwd())
    # Loop through images in current directory.
    for filename in os.listdir():
        inum = inum + 1
        print(inum)
        # Open image.
        try:
            curimg = Image.open(filename)
        except:
            pass
        print(filename)
        curimg = curimg.convert('RGB')
        # Resize image to 1000 pixels wide, maintaining aspect ratio.
        curimg = curimg.resize((1000, int(curimg.size[1]*(1000/curimg.size[0]))))
        # Generate savefile name.
        imgstr = imgnum.filestr(inum)
        # Save into EC602MP2 directory.
        print(todir+imgstr)
        curimg.save(todir + imgstr)

print("Number of images: " + str(imgnum))

