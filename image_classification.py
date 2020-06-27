import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#IMPORT THE FASHION MNIST DATASET
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#class names are not included in the dataset 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#EXPLORE THE DATA
print(train_images.shape)   #60000x28x28
print(len(train_labels))    #60000
print(train_labels)         #integer between 0 and 9
print(test_images.shape)    #10000x28x28
print(len(test_labels))     #10000

#PREPROCESS DATA
#data must be preprocessed before training
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#scale values from (0->255) to (0->1) by dividing 255
train_images = train_images/255.0
test_images = test_images/255.0

#display the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#BUILD THE MODEL
#Set up layers
model = keras.Sequential([
    #transform the format of the images, from a two-dimensional layer
    #28x28 to a one-dimensional array of 28*28=784 pixel
    keras.layers.Flatten(input_shape=(28,28)),
    #2 Dense layers (2 FC nn layers)
    #1st one has 128 nodes (or neurons)
    #2nd one returns a logits array with length of 10
    #to indicate that the current image belongs to which of the 10 classes
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
])

#Compile model
#loss function: measures how accurate the model is (should minimize this)
#optimizer: this is how the model is updated
#metrics: used to minitor the training and testing steps
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model
#Steps:
#(1): Feed the training data to the model
#(2): The model learns to associate images and labels
#(3): Use the test set (test_images)
#(4): Verify that the predictions match the true labels (test_labels)

#(1): Feed the model + (3)
model.fit(train_images, train_labels, epochs=10)
#(3): Evaluate accuracy + (4)
#Compare how the model performs on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
    #if the accuracy on the test set is less than the training set,
    #overfitting happens

#Make predictions
#attach a softmax layer to convert the logits (model's linear output)
#to probabilities, which are easier to interpret
probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#first prediction
#a prediction is an array of 10 numbers, representing the model's confidence
#that the image corresponds to each class
print(predictions[0])
#to see which label has the highest confidence value
print(np.argmax(predictions[0]))

#GRAPHING to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_label,img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        #Correct predictions are blue
        color = 'blue'
    else:
        #Incorrect predictions are red
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#Verify predictions
#0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#12th image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#plot the first 5*3=15 test images
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range (num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#Use the trained model
#Grab an image from the test set
img = test_images[14]
print(img.shape)                    #28x28

#tf.keras models are optimized to make predictions on a batch
#of examples at once. Therefore, even though you are using 1 single image,
#you need to add it to a list
img = (np.expand_dims(img, 0))
print(img.shape)                    #1x28x28

#predict
predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(14, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(test_labels[np.argmax(predictions_single[0])])


