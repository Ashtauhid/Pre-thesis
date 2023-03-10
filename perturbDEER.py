import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image

# Load a new image
new_image = image.load_img('deer.png', target_size=(32, 32))
new_image = image.img_to_array(new_image)

img = mpimg.imread('deer.png')
imgplot = plt.imshow(img)
plt.show()

# Normalize the pixel values
new_image = new_image.astype('float32') / 255.0

# Add a new dimension to match the shape of the training data
new_image = np.expand_dims(new_image, axis=0)