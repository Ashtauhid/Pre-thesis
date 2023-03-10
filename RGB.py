from PIL import Image
from tensorflow.keras.preprocessing import image

# Open the image file
image = Image.open("deer.jpg")

# Load a new image
# new_image = image.load_img('deer.png', target_size=(32, 32))
# image = image.img_to_array(new_image)

# Get the RGB values of each pixel in the image
rgb_image = image.convert('RGB')
width, height = image.size
for x in range(width):
    for y in range(height):
        r, g, b = rgb_image.getpixel((x, y))
        print("Pixel at ({0}, {1}) - Red: {2}, Green: {3}, Blue: {4}".format(x, y, r, g, b))
