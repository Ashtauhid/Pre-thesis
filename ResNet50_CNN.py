import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50

# define the path to the folder containing the training images
train_path = "CIFAR-10/train"

# define the path to the folder containing the test images
test_path = "CIFAR-10/test"

# define the class names and their corresponding integer values
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
class_dict = {class_name: i for i, class_name in enumerate(class_names)}

# define the number of classes in the dataset
num_classes = 10

# define the dimensions of the images
img_rows, img_cols, img_channels = 32, 32, 3

# function to load the images from a folder
def load_images(path):
    images = []
    labels = []
    # iterate over the subdirectories in the folder
    for subdir in os.listdir(path):
        # get the class label for the subdirectory
        label = class_dict[subdir]
        # iterate over the images in the subdirectory
        for file in os.listdir(os.path.join(path, subdir)):
            # load the image using PIL
            img = Image.open(os.path.join(path, subdir, file))
            # resize the image to the desired dimensions
            img = img.resize((img_rows, img_cols))
            # convert the image to a numpy array and normalize its values
            img = np.array(img) / 255.0
            # add the image and label to the lists
            images.append(img)
            labels.append(label)
    # convert the lists to numpy arrays and return them
    return np.array(images), np.array(labels)

# load the training images and labels
train_images, train_labels = load_images(train_path)

# load the test images and labels
test_images, test_labels = load_images(test_path)

# convert the labels to one-hot encoded vectors
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# define (X_train, y_train) and (X_test, y_test)
X_train, y_train = train_images, train_labels
X_test, y_test = test_images, test_labels

# Create ResNet50 model
input_shape = (32, 32, 3)
num_classes = 10
input_tensor = Input(shape=input_shape)
base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, input_shape=input_shape, pooling='avg', classes=num_classes)

# Add top layers to the ResNet50 model
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Set up the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)
batch_size = 64
epochs = 1000
steps_per_epoch = X_train.shape[0] // batch_size
checkpoint = ModelCheckpoint('resnet50_cifar10.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.1)
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, lr_reduction])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

print('Test loss:', loss)
print('Test accuracy:', accuracy)
