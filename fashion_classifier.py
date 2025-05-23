# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Fashion-MNIST dataset
# This dataset contains 60,000 training images and 10,000 test images.
# Each image is 28x28 grayscale pixels and belongs to one of 10 clothing categories.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names (for reference when visualizing)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Explore the data (optional, but recommended)
print("Shape of training images:", train_images.shape)
print("Number of training labels:", len(train_labels))
print("Training labels:", train_labels)
print("Shape of test images:", test_images.shape)
print("Number of test labels:", len(test_labels))

# Visualize an example image from the dataset
# plt.figure()
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.xlabel(class_names[train_labels[0]])
# plt.show()

# 3. Preprocess the data
# Normalize pixel values to be in the range of 0 to 1
# This is done by dividing each pixel value by 255 (the maximum value of a pixel).
train_images = train_images / 255.0
test_images = test_images / 255.0

# The images are 28x28, but a dense neural network expects a flat vector.
# Reshaping is not strictly necessary for a Flatten layer, but it is for a Dense layer as the first layer if Flatten is not used.
# To keep it similar to the MNIST example you had and use Dense layers directly (without an explicit Flatten at the beginning):
train_images_flattened = train_images.reshape((60000, 28 * 28))
test_images_flattened = test_images.reshape((10000, 28 * 28))


# 4. Build the neural network model
# We will use a sequential model, similar to the MNIST example.
model = models.Sequential([
    # First dense (fully connected) layer with 128 neurons and ReLU activation.
    # input_shape is only necessary in the first layer if the model is not built with `model.build()` or trained before.
    # As an alternative to reshape, we could use layers.Flatten(input_shape=(28, 28)) as the first layer.
    layers.Dense(128, activation='relu', input_shape=(28 * 28,)), # 784 input neurons (28*28)
    # Output layer with 10 neurons (one for each clothing class) and softmax activation.
    # Softmax converts the outputs into a set of probabilities that sum to 1.
    layers.Dense(10, activation='softmax')
])

# 5. Compile the model
# Here we configure the learning process.
model.compile(optimizer='adam', # Adam optimizer, a common alternative to rmsprop
              loss='sparse_categorical_crossentropy', # Suitable for integer labels
              metrics=['accuracy']) # Metric to monitor: accuracy

# Print a summary of the model
model.summary()

# 6. Train the model
# We fit the model using the training data.
print("\n--- Training the model ---")
history = model.fit(train_images_flattened,
                    train_labels,
                    epochs=10, # Number of times the model will see the entire dataset
                    batch_size=64, # Number of images to process in each step before updating weights
                    validation_split=0.1) # Use 10% of the training data for validation

# 7. Evaluate the model
# We check how well the model performs on data it has never seen (the test set).
print("\n--- Evaluating the model ---")
test_loss, test_acc = model.evaluate(test_images_flattened, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

# 8. Make predictions
print("\n--- Making predictions ---")
predictions = model.predict(test_images_flattened)

# The output of 'predict' is an array of arrays, where each inner array has 10 numbers.
# These numbers represent the model's "confidence" that the image corresponds to each of the 10 articles of clothing.
# We can take the index with the highest value as the model's prediction.
print("Prediction for the first test image:", predictions[0])
print("The predicted label is:", np.argmax(predictions[0]))
print("The true label is:", test_labels[0])
print("Predicted class name:", class_names[np.argmax(predictions[0])])
print("True class name:", class_names[test_labels[0]])


# 9. Visualize some predictions (optional)
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i] # Use original 2D images for visualization
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue' # Correct prediction
  else:
    color = 'red' # Incorrect prediction
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Visualize the i-th image, its prediction, and the probability array.
# For example, for the first 15 images:
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  # Pass test_images (original 2D) for plot_image,
  # and predictions[i] (which was calculated with test_images_flattened)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# You can also save the model if you want to use it later without retraining
# model.save('fashion_mnist_model.h5')
# To load it:
# loaded_model = models.load_model('fashion_mnist_model.h5')
