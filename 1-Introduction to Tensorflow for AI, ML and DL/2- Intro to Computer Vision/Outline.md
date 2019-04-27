## Week 2 Intro to Computer Vision

#### I. Computer Vision

It's hard to set rules for computer vision, say to write code to tell is this a dress, a pair of pants, or a pair of shoes (through pixel), etc. So the labeled samples are the way to go.

How would you program shirts, shoes, clothes, etc?

One way to solve that is to use lots of pictures of clothing and tell the computer __what that's a picture of__ and then __have the computer figure out the patterns__ that give you the difference between a shoe, and a shirt, and a handbag, and a coat. That's what you're going to learn how to do in this section. 

Fortunately, there's a data set called Fashion MNIST which gives a 70 thousand images spread across 10 different items of clothing. 

![MNIST](https://raw.githubusercontent.com/JiaRuiShao/TensorFlow/master/1-Introduction%20to%20Tensorflow%20for%20AI%2C%20ML%20and%20DL/images/MNIST.PNG)

These images have been scaled down to 28 by 28 pixels. Now usually, the smaller the better because the computer has less processing to do. But of course, you need to retain enough information to be sure that the features and the object can still be distinguished. 

If you look at these pictures you can still tell the difference between shirts, shoes, and handbags. So this size does seem to be ideal, and it makes it great for training a neural network. The images are also in gray scale, so the amount of information is also reduced. Each pixel can be represented in values from zero to 255 and so it's only one byte per pixel. With 28 by 28 pixels in an image, only 784 bytes are needed to store the entire image. 

#### II. Load Training Data


```python
fashion_mnist =  tf.keras.datasets.fashion_mnist
```

On this object, if we call the load data method, it will return four lists to us. That's the training data, the training labels, the testing data, and the testing labels. 

When building a neural network like this, it's a nice strategy to use some of your data to train the neural network and similar data that the model hasn't yet seen to test how good it is at recognizing the images. 

So in the Fashion-MNIST data set, 60,000 of the 70,000 images are used to train the network, and then 10,000 images, one that it hasn't previously seen, can be used to test just how good or how bad it is performing. 

```python
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

So this code above will give you those sets. Then, each set has data, the images themselves and labels and that's what the image is actually of. 

![W2.1](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W2.1.PNG?raw=true)

So for example, the training data will contain images like this one, and a label that describes the image like this. While this image is an ankle boot, the label describing it is the number nine. 

There're two main reasons that the lables are numbers: 

- First, of course, is that computers do better with numbers than they do with texts. 
- Second, importantly, is that this is something that can help us reduce bias.

#### III. Machine Learning Fairness

[Click to watch this video](https://developers.google.com/machine-learning/fairness-overview/)

With traditional programming, people hand-code the solution to a problem, step by step. With ML, computers learn the solution by finding pattens in data.

Our human biases become part of the technology we created in many different ways.

- interaction bias
- latent bias
- selction bias

#### IV. Coding a Computer Vision Neural Network

```python
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), #Flatten takes this 28 by 28 square and turns it into a simple linear array(image shape 28x28)
	keras.layers.Dense(128, activation=tf.nn.relu), # input num is 128 
	keras.layers.Dense(10,activation=tf.nn.softmax) # output num is 10(10 neurons here because we have ten classes of clothing in the dataset)
	])
```

__Structure__

- The first line of code is the input layer
- The second line of code is the hidden layer
- The third line of code is the output layer

#### V. Walk through a Notebook for computer vision

[Example](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)

When you’re done with that, the next thing to do is to explore callbacks, so you can see how to train a neural network until it reaches a threshold you want, and then stop training. 

#### VI. Using Callbacks to Control Training

How can I stop training when I reach a point that I want to be at?

In every epoch, you can callback to a code function, having checked the metrics. If they're what you want to say, then you can cancel the training at that point. 

```python
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), 
(test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

# execute the training loop
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

```

Implement `callback` function:

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}): # we focused on the num at the end of epoch
    if(logs.get('acc')>0.6): # or we could say: if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
```

Two modifications to make:

```python
callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), 
(test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

# execute the training loop
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

```

[callbacks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb)

Bring all these steps together to create a neural network model using TensorFlow:

```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

```

### Weekly Exercise -- Implement a Deep Neural Network to recognize handwritten digits

[Exercise](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%202%20-%20Handwriting%20Recognition/Exercise2-Question.ipynb)

[Answer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%202%20-%20Handwriting%20Recognition/Exercise2-Answer.ipynb)



