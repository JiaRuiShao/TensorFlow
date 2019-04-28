## Week 4 Using Real-World Images

__TensorFlow Using Scenario__:

[TensorFlow: an ML platform for solving impactful and challenging problems](https://www.youtube.com/watch?v=NlpS-DhayQA)

#### I. Understanding ImageGenerator

To this point, you built an image classifier that worked using a deep neural network and you saw how to improve its performance by adding convolutions.

**Limitations**

One limitation though was that it used a dataset of very uniform images. Images of clothing that was staged and framed in 28 by 28. But what happens when you use larger images and where the feature might be in different locations?

In addition to that, the earlier examples with a fashion data used a built-in dataset. All of the data was handily split into training and test sets for you and labels were available. In many scenarios, that's not going to be the case and you'll have to do it for yourself.

In this lesson, we'll take a look at some of the APIs that are available to make that easier for you. In particular, the __image generator__ in TensorFlow.

**Features of ImageGenerator**:

- labels of sub-directories will be automatically generate after you point at a directory

![W4.1](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.1.PNG?raw=true)

For example, consider this directory structure. You have an images directory and in that, you have sub-directories for training and validation. When you put sub-directories in these for horses and humans and store the requisite images in there, the image generator can create a feeder for those images and auto label them for you. 

![W4.2](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.2.PNG?raw=true)

For example, if I point an image generator at the training directory, the labels will be horses and humans and all of the images in each directory will be loaded and labeled accordingly. 

![W4.3](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.3.PNG?raw=true)

Similarly, if I point one at the validation directory, the same thing will happen.


**Image Generator**

```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator
```

The image generator class is available in Keras.preprocessing.image. 

```python
train_datagen = ImageDataGenerator(rescale=1./255) #  rescale to it to normalize the data

train_generator = train_datagen.flow_from_directory(# call the flow from directory method on it to get it to load images from that directory and its sub-directories

	train_dir, # directory

	target_size=(300,300), # used to resize the images to make them consistent before process

	batch_size=128, # The images will be loaded for training and validation in batches where it's more efficient than doing it one by one. The batch size of training dataset is 128.

	class_mode='binary') # This is a binary classifier, horse and human
```

Then instantiate an image generator like this. I'm going to pass rescale to it to normalize the data. You can then call the flow from directory method on it to get it to load images from that directory and its sub-directories. 

Note: It's a common mistake that people point the generator at the sub-directory(The names of the sub-directories will be the labels for your images that are contained within them). You should always point it at the directory that contains sub-directories that contain your images. 

The validation generator should be exactly the same except of course it points at a different directory, the one containing the sub-directories containing the test images.

**Validation Generator**

The validation generator should be exactly the same except of course it points at a different directory, the one containing the sub-directories containing the test images.

```python
test_datagen = ImageDataGenerator(rescale=1./255) 
# rescale to it to normalize the data

test_generator = test_datagen.flow_from_directory(
# call the flow from directory method on it to get it to load images from that directory and its sub-directories

	validation_dir, # validation directory

	target_size=(300,300), # used to resize the images to make them consistent before process

	batch_size=32, # The batch size of training dataset is 32.

	class_mode='binary') # This is a binary classifier, horse and human
```

Now that you’ve seen how an ImageGenerator can flow images from a directory and perform operations such as resizing them on the fly, the next thing to do is design the neural network to handle these more complex images.

#### II. Defining a ConvNet to use complex images

The code that we'll use to classify horses versus humans is very similar to what you just used for the fashion items, but there are a few minor differences based on this data, and the fact that we're using generators.

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'), # hidden layer
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```

First of all, you'll notice that there are three sets of convolution pooling layers at the top. This reflects the higher complexity and size of the images. We can even add another couple of layers to this if we wanted to get to the same ballpark size as previously, but we'll keep it at three for now.

Another thing to pay attention to is the input shape. We resize their images to be 300 by 300 as they were loaded, but they're also color images. So there are three bytes per pixel. One byte for the red, one for green, and one for the blue channel, and that's a common 24-bit color pattern.

The output layer has also changed. Remember before when you created the output layer, you had one neuron per class, but now there's only one neuron for two classes.

That's because we're using a different activation function where __sigmoid is great for binary classification__, where one class will tend towards zero and the other class tending towards one. 

You could use two neurons here if you want, and the same softmax function as before, but for binary this is a bit more efficient. If you want you can experiment with the workbook and give it a try yourself. 

Now, if we take a look at our model summary, we can see the journey of the image data through the convolutions.

![W4.4](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.4.PNG?raw=true)

Now that you’ve designed the neural network to classify Horses or Humans, the next step is to train it from data that’s on the file system, which can be read by generators. To do this, you don’t use model.fit as earlier, but a new method call: __model.fit_generator__.

#### III. Training the ConvNet with fit_generator

__Compile the model__:(loss function + optimizer)

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', 
			  optimizer=RMSprop(1r=0.001), 
			  # use rmsprop optimizer with a learning rate of 0.001
			  metrics=['acc'])
```

You might remember that your loss function was a __categorical cross entropy__used for multi-class classification. But because we're doing a binary choice here, let's pick a __binary_crossentropy__ instead.

Also, earlier we used an __Adam optimizer__. Now, you could do that again, but I thought it would be fun to use the __RMSprop__, where you can __adjust the learning rate to experiment with performance__.

**NOTE**: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)

__Train the model__:

```python
history = model.fit_generator(
	train_generator, 
	# the training generator that you set up earlier

	steps_per_epoch=8, 
	# 8 = the num of training example/training batch size = 1024/128

	epochs=15, # set the number of epochs to train for

	validation_data=validation_generator,
	# specify the validation set that comes from the validation_generator that we created earlier

	validation_steps=8, # 8 = the num of testing example/testing batch size = 256/32

	verbose=2) # how much to display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress.
```

The first parameter is the training generator that you set up earlier. This streams the images from the training directory.

Remember the batch size you used when you created it, it was 20, that's important in the next step. There are 1,024 images in the training directory, so we're loading them in 128 at a time. So in order to load them all, we need to do 8 batches. So we set the steps_per_epoch to cover that.

__Predict using the built model__

![W4.5](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.5.PNG?raw=true)

These parts are specific to Colab, they are what gives you the button that you can press to pick one or more images to upload. The image paths then get loaded into this list called uploaded. 

![W4.6](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.6.PNG?raw=true)

The loop then iterates through all of the images in that collection. And you can load an image and prepare it to input into the model with this code. Take note to ensure that the dimensions match the input dimensions that you specified when designing the model. 

![W4.7](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.7.PNG?raw=true)

You can then call model.predict, passing it the details, and it will return an array of classes. __In the case of binary classification, this will only contain one item with a value close to 0 for one class and close to 1 for the other__.

Later in this course you'll see __multi-class classification with Softmax Where you'll get a list of values with one value for the probability of each class and__ all of the probabilities adding up to 1.

You’ve gone through the code to define the neural network, train it with on-disk images, and then predict values for new images. Let’s see this in action in a workbook. 

More about [Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/), [RMSProp Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer), and [learning	methods	for	neural	networks](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

#### IV. Hands-on Exercise -- Horse-or-Human-NoValidation

[Example - Horse-or-Human-NoValidation](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb)

#### V. Validate your training model

[Example - Horse-or-Human-WithValidation](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)

#### VI. Compress your images

The images in the horses are humans dataset are all 300 by 300 pixels. So we had quite a few convolutional layers to reduce the images down to condensed features. Now, this of course can slow down the training. So let's take a look at what would happen if we change it to a 150 by a 150 for the images to have a quarter of the overall data and to see what the impact would be.

Note: the target_size of training data and validation data are now changed to 150x150.

__Build the model__

```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    
    # The fifth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

__Data preprocessing__

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow testing images in batches of 32 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

__Predict using the model__

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
 
```

	Pros:

- take less time to train

	Cons:

- might lose featured data

__Now try it on your own!__

Experiment with different sizes -- you don’t have to use 150x150 for example!

[Exercise -- Horse-or-Human-try-compress-data](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb)

#### VII. Quiz

1. Using Image Generator, how do you label images?

	It’s based on the directory the image is contained in

2. What method on the Image Generator is used to normalize the image?

	rescale

3. How did we specify the training size for the images?

	The target_size parameter on the training generator

4. When we specify the input_shape to be (300, 300, 3), what does that mean?

	Every Image will be 300x300 pixels, with 3 bytes to define color

5. If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?

	You’re overfitting on your training data

6. Convolutional Neural Networks are better for classifying images like horses and humans because:

	- In these images, the features may be in different parts of the frame

	- There’s a wide variety of horses and humans

7. After reducing the size of the images, the training results were different. Why?

	We removed some convolutions to handle the smaller images

### Weekly Exercise -- Handling complex images

Let’s now create your own image classifier for complex images. See if you can create a classifier for a set of happy or sad images that I’ve provided. Use a callback to cancel training once accuracy is greater than .999.

[Exercise 2 -- Happy-or-Sad](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%204%20-%20Handling%20Complex%20Images/Exercise%204-Question.ipynb)

[Exercise 2 Answer -- Happy-or-Sad](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%204%20-%20Handling%20Complex%20Images/Exercise4-Answer.ipynb)

__More about batch size__:

The batch size is the number of training samples your training will use in order to make one update to the model parameters. Ideally you would use all the training samples to calculate the gradients for every single update, however that is not efficient. The batch size simply put, will simplify the process of updating the parameters.


![W4.8](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W4.8.PNG?raw=true)

The plot above shows the effect of the batch size on the validation accuracy of the model. One can easily see that the batch size, which contributes heavily in determining the learning parameters, will affect the prediction accuracy.
