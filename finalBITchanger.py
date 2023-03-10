# This is a test model where I try to change the 3 LSBs from all the train images

import os
import numpy as np
from PIL import Image

# Specify the directory containing the images
directory = 'CIFAR-10/train/truck/'

# Loop over all the files in the directory
for filename in os.listdir(directory):
    # Load the image from file
    img = Image.open(os.path.join(directory, filename))

    # Convert the image to a numpy array
    arr = np.array(img)

    # Flip the 3 LSBs of each pixel in the array
    arr = np.bitwise_xor(arr, 7)

    # Convert the array back to an image
    img = Image.fromarray(arr)

    # Save the modified image to a new file
    output_filename = os.path.join(directory, 'flipped_' + filename)
    img.save(output_filename)
