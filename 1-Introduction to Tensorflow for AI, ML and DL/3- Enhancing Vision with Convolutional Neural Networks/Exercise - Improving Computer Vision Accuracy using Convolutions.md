
# Improving Computer Vision Accuracy using Convolutions

In the previous lessons you saw how to do fashion recognition using a Deep Neural Network (DNN) containing three layers -- the input layer (in the shape of the data), the output layer (in the shape of the desired output) and a hidden layer. You experimented with the impact of different sized of hidden layer, number of training epochs etc on the final accuracy.

For convenience, here's the entire code again. Run it and take a note of the test accuracy that is printed out at the end. 


```python
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images / 255.0
test_images=test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)
```

Your accuracy is probably about 89% on training and 87% on validation...not bad...But __how do you make that even better? One way is to use something called Convolutions__. I'm not going to details on Convolutions here, but the ultimate concept is that they __narrow down the content of the image to focus on specific, distinct, details__. 

If you've ever done image processing using a filter (like this: https://en.wikipedia.org/wiki/Kernel_(image_processing)) then convolutions will look very familiar.

In short, you take an filter (usually 3x3 or 5x5) and pass it over the image. By changing the underlying pixels based on the formula within that matrix, you can do things like edge detection. So, for example, if you look at the above link, you'll see a 3x3 that is defined for edge detection where the middle cell is 8, and all of its neighbors are -1. In this case, for each pixel, you would multiply its value by 8, then subtract the value of each neighbor. __Do this for every pixel, and you'll end up with a new image that has the edges enhanced__.

This is perfect for computer vision, because often it's features that can get highlighted like this that distinguish one item for another, and __the amount of information needed is then much less because you'll just train on the highlighted features__.

That's the concept of Convolutional Neural Networks. Add some layers to do convolution before you have the dense layers, and then __the information going to the dense layers is more focussed, and possibly more accurate__.

Run the below code -- this is the same neural network as earlier, but this time with Convolutional layers added first. It will take longer, but look at the impact on the accuracy:


```python
import tensorflow as tf
print(tf.__version__)

# load the data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# reshape & standardize the training data
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

# reshape & standardize the testing data
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# set the neural network model with 2 convolutional layers, and 1 hidden layer
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'), # hidden layer
  tf.keras.layers.Dense(10, activation='softmax')
])

# set the rules to optimize the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# return the size and the shape of the model
model.summary()

# train the model
model.fit(training_images, training_labels, epochs=5)

# evaluate the model on testing data
test_loss = model.evaluate(test_images, test_labels)

```

It's likely gone up to about 93% on the training data and 91% on the validation data. 

That's significant, and a step in the right direction!

Try running it for more epochs -- say about 20, and explore the results! But while the results might seem really good, the validation results may actually go down, due to something called 'overfitting' which will be discussed later. 

(In a nutshell, 'overfitting' occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at seeing *other* data. For example, if all your life you only saw red shoes, then when you see a red shoe you would be very good at identifying it, but blue suade shoes might confuse you...and you know you should never mess with my blue suede shoes.)

<font color = "green">
## Look at the code again, and __see step by step how the Convolutions were built__:

### Step 1 is to load the data. 

You'll notice that there's a bit of a change here in that __the training data needed to be reshaped before the convolutions__. That's because the first convolution expects a single tensor containing everything, so instead of 60,000 28x28x1 items in a list, we have a __single 4D list that is 60,000x28x28x1__, and the same for the test images. If you don't do this, you'll get an error when training as the Convolutions do not recognize the shape. 



```python
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
```



### Step 2 is to define your model. 

Now instead of the input layer at the top, you're going to add a Convolution. The parameters are:

1. __The number of convolutions you want to generate__. Purely arbitrary, but good to start with something in the order of 32
2. __The size of the Convolution__, in this case a 3x3 grid
3. __The activation function to use__ -- in this case we'll use relu, which you might recall is the equivalent of returning x when x>0, else returning 0
4. __The shape of the input data in the first layer__.

You'll follow the Convolution with a __MaxPooling layer__ which is then designed to __compress the image__, while maintaining the content of the features that were highlighted by the convlution. By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image. Without going into too much detail here, the idea is that __it creates a 2x2 array of pixels, and picks the biggest one, thus turning 4 pixels into 1__. It repeats this across the image, and in so doing __halves the number of horizontal, and halves the number of vertical pixels__, effectively reducing the image by 25%/a quarter.

You can call __model.summary()__ to see the __size and shape of the network__, and you'll notice that after every MaxPooling layer, the image size is reduced in this way. 


```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
```



Add another convolution



```python
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2)
```



Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version

```python
  tf.keras.layers.Flatten(),
```



The same 128 dense layers, and 10 output layers as in the pre-convolution example:



```python
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
```



Now compile the model, call the fit method to do the training, and evaluate the loss and accuracy from the test set.



```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```




# Visualizing the Convolutions and Pooling

This code will show us the convolutions graphically. The print (test_labels[;100]) shows us the first 100 labels in the test set, and you can see that the ones at index 0, index 23 and index 28 are all the same value (9). They're all shoes. Let's take a look at the result of running the convolution on each, and you'll begin to see common features between them emerge. Now, when the DNN(Deep Neural Network) is training on that data, it's working with a lot less, and it's perhaps finding a commonality between shoes based on this convolution/pooling combination.


```python
print(test_labels[:100])
```


```python
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=3
SECOND_IMAGE=55
THIRD_IMAGE=11
CONVOLUTION_NUMBER = 32
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
```

### EXERCISES

1. Try editing the convolutions. Change the 32s to either 16 or 64. What impact will this have on accuracy and/or training time.

2. Remove the final Convolution. What impact will this have on accuracy or training time?

3. How about adding more Convolutions? What impact do you think this will have? Experiment with it.

4. Remove all Convolutions but the first. What impact do you think this will have? Experiment with it. 

5. In the previous lesson you implemented a callback to check on the loss function and to cancel training once it hit a certain amount. See if you can implement that here!


```python
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```

Q1. 

convolution # |  16  |  32  |  64
--------------|------|------|------
training acc  |0.9979|0.9985|0.9983    
testing acc   |0.9872|0.9862|0.9863    
time to train |394s  |672s  |1232s


```python
import tensorflow as tf
print(tf.__version__)

# gather the data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# reshape & standardize the training data
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0

# reshape & standardize the testing data
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# change the variables here
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
    
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'), # hidden layer
  tf.keras.layers.Dense(10, activation='softmax')
])

# set the rules to optimize the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# return the size and the shape of the model
model.summary()

# train the model
model.fit(training_images, training_labels, epochs=5)

# evaluate the model on testing data
test_loss = model.evaluate(test_images, test_labels)

```

    1.13.1
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 11, 11, 32)        9248      
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 5, 5, 32)          0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 3, 3, 32)          9248      
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 1, 1, 32)          0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 32)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 128)               4224      
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 24,330
    Trainable params: 24,330
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/5
    60000/60000 [==============================] - 70s 1ms/sample - loss: 0.6531 - acc: 0.7563
    Epoch 2/5
    60000/60000 [==============================] - 71s 1ms/sample - loss: 0.4459 - acc: 0.8369
    Epoch 3/5
    60000/60000 [==============================] - 72s 1ms/sample - loss: 0.3873 - acc: 0.8587
    Epoch 4/5
    60000/60000 [==============================] - 70s 1ms/sample - loss: 0.3497 - acc: 0.8707
    Epoch 5/5
    60000/60000 [==============================] - 66s 1ms/sample - loss: 0.3267 - acc: 0.8792
    10000/10000 [==============================] - 4s 391us/sample - loss: 0.3502 - acc: 0.8742
    

Q2&3&4.

Q2&3&4       | remove all but the first | remove final Convolution | not remove final Convolution | add one more Convolution
-------------|--------------------------|--------------------------|------------------------------|-------------------------     
training acc |0.8524                    |0.9403                    |0.9187                        |0.8792 
testing acc  |0.8289                    |0.9107                    |0.9041                        |0.8742
training time|92s                       |365s                      |(forgot lol)                  |349s


```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```

    Epoch 1/5
    60000/60000 [==============================] - 42s 694us/sample - loss: 0.1680 - acc: 0.9514
    Epoch 2/5
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0554 - acc: 0.9836
    Reached 98% accuracy so cancelling training!
    60000/60000 [==============================] - 36s 592us/sample - loss: 0.0554 - acc: 0.9836
    10000/10000 [==============================] - 2s 217us/sample - loss: 0.0556 - acc: 0.9818
    0.9818
    
