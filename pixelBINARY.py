# from PIL import Image
#
# # Load the image file
# img = Image.open('deer.jpg')
#
# # Get the dimensions of the image
# width, height = img.size
# # Loop through each pixel and convert its value to binary
# for y in range(height):
#     for x in range(width):
#         pixel = img.getpixel((x, y))
#         r, g, b = pixel
#         r_bin = bin(r)[2:].zfill(8)
#         g_bin = bin(g)[2:].zfill(8)
#         b_bin = bin(b)[2:].zfill(8)
#         binary = r_bin + g_bin + b_bin
#         print(binary)
#

from PIL import Image

# Load the image file
img = Image.open('deer.jpg')

# Get the dimensions of the image
width, height = img.size

# Loop through each pixel and flip the last 4 bits of its value
for y in range(height):
    for x in range(width):
        pixel = list(img.getpixel((x, y)))
        for i in range(3):  # iterate over the 3 color channels
            binary = bin(pixel[i])[2:].zfill(8)
            binary = binary[:-4] + ''.join([str(1 - int(bit)) for bit in binary[-4:]])  # flip the last 4 LSBs
            pixel[i] = int(binary, 2)
        img.putpixel((x, y), tuple(pixel))

# Save the modified image
img.save('modified_image.jpg')


# from PIL import Image
#
# # Load the image file
# img = Image.open('deer.jpg')
#
# # Get the dimensions of the image
# width, height = img.size
#
# # Define the trigger message to be hidden
# trigger_message = "This is a trigger message."
#
# # Convert the trigger message to binary
# trigger_binary = ''.join(format(ord(c), '08b') for c in trigger_message)
#
# # Insert the trigger binary into the LSBs of the image pixels
# index = 0
# for y in range(height):
#     for x in range(width):
#         if index >= len(trigger_binary):  # check if all trigger bits have been inserted
#             break
#         pixel = list(img.getpixel((x, y)))
#         for i in range(3):  # iterate over the 3 color channels
#             binary = bin(pixel[i])[2:].zfill(8)
#             binary = binary[:-1] + trigger_binary[index]  # replace the LSB with the trigger binary digit
#             pixel[i] = int(binary, 2)
#             index += 1
#             if index == len(trigger_binary):  # stop if all trigger bits have been inserted
#                 break
#         img.putpixel((x, y), tuple(pixel))
#     if index == len(trigger_binary):  # stop if all trigger bits have been inserted
#         break
#
# # Save the modified image
# img.save('modified_image.jpg')
