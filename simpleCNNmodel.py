# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
#
# # Load the CIFAR10 dataset
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#
# # Normalize the data
# X_train = X_train.astype('float32') / 255.0
# X_test = X_test.astype('float32') / 255.0
#

import numpy as np
from tensorflow.keras.preprocessing import image

# Load a new image
new_image = image.load_img('modified_image.jpg', target_size=(32, 32))
new_image = image.img_to_array(new_image)

# Normalize the pixel values
new_image = new_image.astype('float32') / 255.0

# Add a new dimension to match the shape of the training data
new_image = np.expand_dims(new_image, axis=0)


# # Convert the labels to one-hot encoded vectors
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
#
# # Define the model
# model = Sequential()
#
# # Add convolutional layers
# model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32,32,3)))
# model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(rate=0.25))
#
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(rate=0.25))
#
# # Add fully connected layers
# model.add(Flatten())
# model.add(Dense(units=512, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(units=10, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
#
# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)

# Save the trained model
# model.save('cifar10_model.h5')

# Load the saved model
model = load_model('cifar10_model.h5')

# Make predictions on new images
predictions = model.predict(new_image)
print("The class of the new image is:", np.argmax(predictions, axis=1))