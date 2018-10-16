# EC601MP2
Deep learning project to differentiate between images of two different types of objects.

# ImageFixer
This python script was used to pull and compress images from my personal collection. There are some gaps in the file numbers due to a few hundred of the pictures being ill suited for this project. I manually deleted them after the batch pull. imagefixer.py does not need to be rerun, but is left in the folder for documentation purposes. Since the write-to folder has been removed, it will not work if atttempted.

# ImageBW200

After manually cropping the photos to focus on the desired subject and give them a standard aspect ratio, I wrote this script to reduce their size to a uniform 300x200 and convert them to black and white since we're focusing on differentiating between shapes and color doesn't matter. The script also randomly assigned the pictures to training and test data with an 80:20 split. After this operation was completed, the ImagesW1K was moved out of the git path to reduce download size. This also doesn't need to be rerun and is left for documentation. Since the input folder has been moved, it will not work.

# LabelCreator

This script runs through both train and test folders and asks the user if the corresponding image is an airplane. The user should input 1 if yes, 0 if no. This script will create and save a 3D array of images and a corresponding 1D array of lables. It does not need to be rerun but will work if attempted.

# LayerTester

This script uses a nested loop to create models with 1-10 dense layers with 32-1024 nodes (node count doubled each iteration). It then prints the resulting accuracy sorted by nodecount, then layer count in ascending order. This data is used to generate the test values for the next script. 

# EpochTester

This script tests the best layer/node count values from the LayerTester to figure out how many Epoch's are ideal. The Epoch count ranges from 5-20. It will then print the resulting accuracy. The layer/node/epoch cout result will be used in the main program.
