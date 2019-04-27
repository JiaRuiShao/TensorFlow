## Week 3 Enhancing Vision with Convolutional Neural Networks

#### I. Convolutions and pooling layers

Now, one of the things that you would have seen when you looked at the images is that there's a lot of wasted space in each image. While there are only 784 pixels, it will be interesting to see if there was a way that we could condense the image down to the important features that distinguish what makes it a shoe, or a handbag, or a shirt. That's where convolutions come in. 

So, what's __convolution__? 

some convolutions will change the image in such a way that certain features in the image get emphasized. 

__Pooling__ is a way of compressing an image. A quick and easy way to do this, is to go over the image of four pixels at a time, i.e, the current pixel and its neighbors underneath and to the right of it. Of these four, pick the biggest value and keep just that.

Convolutions are really powerful when combined with pooling.


The concepts introduced here are available as [Conv2D layers](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/keras/layers/Conv2D) and [MaxPooling2D layers](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling2D) in TensorFlow. 

#### II. Implementing convolutional layers

Here's our code from the earlier example, where we defined out a neural network to have an input layer in the shape of our data, and output layer in the shape of the number of categories we're trying to define, and a hidden layer in the middle. The Flatten takes our square 28 by 28 images and turns them into a one dimensional array.

```python
model = tf.keras.models.Sequential([
	tf.kears.layers.Flattern()
	tf.keras.layers.Dense(128, activation = tf.nn.relu)
	tf.keras.layers.Dense(10, activation = tf.nn.softmax)
	])
```


To add convolutions to this, you use code like this. 

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

You'll see that the last three lines are the same, the Flatten, the Dense hidden layer with 128 neurons, and the Dense output layer with 10 neurons. What's different is what has been added on top of this. Let's take a look at this, line by line.

```python
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))
```

Here we're specifying the first convolution. We're asking keras to generate 64 filters for us. These filters are 3 by 3, their activation is relu, which means the negative values will be thrown way, and finally the input shape is as before, the 28 by 28. That extra 1 just means that we are tallying using a single byte for color depth. As we saw before our image is our gray scale, so we just use one byte.

You might wonder what the 64 filters are. It's a little beyond the scope of this class to define them, but they aren't random. They start with a set of known good filters in a similar way to the pattern fitting that you saw earlier, and the ones that work from that set are learned over time.

For more details on convolutions and how they work, there's a great set of resources here ([Course 4 in DL Specialization -- Convolutional Neural Networks](https://github.com/JiaRuiShao/Deep-Learning/tree/DL/Convolutional%20Neural%20Networks)).


#### III. Implementing pooling layers

This next line of code will then create a pooling layer. 

```python
tf.keras.layers.MaxPooling2D(2, 2)
```

It's max-pooling because we're going to take the maximum value. We're saying it's a two-by-two pool, so for every four pixels, the biggest one will survive as shown earlier. 

```python
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2)
```

We then add another convolutional layer, and another max-pooling layer so that the network can learn another set of convolutions on top of the existing one, and then again, pool to reduce the size. 

By the time the image gets to the flatten to go into the dense layers, it's already much smaller. It's being quartered, and then quartered again. 

So, its content has been greatly simplified, the goal being that __the convolutions will filter it to the features that determine the output__. 

A really useful method on the model is the `model.summary()` method. This allows you to inspect the layers of the model, and see the journey of the image through the convolutions, and here is the output. 

![W3.1]()

It's a nice table showing us the layers, and some details about them including the output shape. It's important to keep an eye on the output shape column. When you first look at this, it can be a little bit confusing and feel like a bug. 

After all, isn't the data 28 by 28, so y is the output, 26 by 26. The key to this is remembering that the filter is a three by three filter. 

Consider what happens when you start scanning through an image starting on the top left. So, for example with this image of the dog on the right, you can see zoomed into the pixels at its top left corner. 

![W3.2]()

You can't calculate the filter for the pixel in the top left, because it doesn't have any neighbors above it or to its left. 

![W3.3]()

In a similar fashion, the next pixel to the right won't work either because it doesn't have any neighbors above it. So, logically, the first pixel that you can do calculations on is this one, 

![W3.5]()

because this one of course has all eight neighbors that a three by three filter needs. 


This means that you can't use a one pixel margin all around the image, so the output of the convolution will be two pixels smaller on x, and two pixels smaller on y. 

If your filter is five-by-five for similar reasons, your output will be four smaller on x, and four smaller on y. 

So, that's y with a three by three filter, our output from the 28 by 28 image, is now 26 by 26, we've removed that one pixel on x and y, and each of the borders. 

Next is the first of the max-pooling layers. 

Now, remember we specified it to be two-by-two, thus turning four pixels into one, and having our x and y. Now our output gets reduced from 26 by 26, to 13 by 13. 

![W3.6]()

The convolutions will then operate on that, and of course, we lose the one pixel margin as before, so we're down to 11 by 11, 

![W3.7]()

add another two-by-two max-pooling to have this rounding down, and went down, down to five-by-five images.

![W3.8]()

So, now our dense neural network is the same as before, but it's being fed with five-by-five images instead of 28 by 28 ones. 

But remember, it's not just one compress five-by-five image instead of the original 28 by 28, there are __a number of convolutions per image that we specified, in this case 64__. So, there are 64 new images of five-by-five that had been fed in. 

Flatten that out and you have 25 pixels times 64, which is 1600. 

![W3.9]()

So, you can see that the new flattened layer has 1,600 elements in it, as opposed to the 784(28*28) that you had previously. 

__This number is impacted by the parameters that you set when defining the convolutional 2D layers__. Later when you experiment, you'll see what the impact of setting what other values for the number of convolutions will be, and in particular, you can see what happens when you're feeding less than 784 over all pixels in. 

Training should be faster, but is there a sweet spot where it's more accurate? Well, let's switch to the workbook, and we can try it out for ourselves.

**Summary**

You’ve now seen how to turn your Deep Neural Network into a Convolutional Neural Network by adding convolutional layers on top, and having the network train against the results of the convolutions instead of the raw pixels.

#### Improving the Fashion classifier with convolutions

You've looked at convolutions and got a glimpse for how they worked. By __passing filters over an image to reduce the amount of information__, they then allowed the neural network to __effectively extract features that can distinguish one class of image from another__. You also saw how __pooling compresses the information to make it more manageable__.

#### Hands-on Exercise

[Here’s](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb) the notebook that Laurence was using in that screencast. To make it work quicker, go to the ‘Runtime’ menu, and select ‘Change runtime type’. Then select GPU as the hardware accelerator.

[play with convolutions](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)

Try different filters, and research different filter types. There's some fun information about filters [here](https://lodev.org/cgtutor/filtering.html).

There are a few rules about the filter:
- Its size has to be uneven, so that it has a center, for example 3x3, 5x5 and 7x7 are ok.
- It doesn't have to, but the sum of all elements of the filter should be 1 if you want the resulting image to have the same brightness as the original.
- If the sum of the elements is larger than 1, the result will be a brighter image, and if it's smaller than 1, a darker image. If the sum is 0, the resulting image isn't necessarily completely black, but it'll be very dark.

Apart from using a filter matrix, it also has a multiplier factor and a bias. After applying the filter, the factor will be multiplied with the result, and the bias added to it. So if you have a filter with an element 0.25 in it, but the factor is set to 2, all elements of the filter are in theory multiplied by two so that element 0.25 is actually 0.5. The bias can be used if you want to make the resulting image brighter.

#### Quiz

1. What is a Convolution?

A technique to isolate features in images

2. What is a Pooling?

A technique to reduce the information in an image while maintaining features

3. How do Convolutions improve image recognition?

They isolate features in images

4. After passing a 3x3 filter over a 28x28 image, how big will the output be?

26x26

5. After max pooling a 26x26 image with a 2x2 filter, how big will the output be?

13x13

6. Applying Convolutions on top of our Deep neural network will make training:

It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!

### Weekly Exercise - Improving DNN Performance using Convolutions

[Exercise 3 - Improve MNIST with convolutions](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Question.ipynb)

[Exercise 3 Answer - Improve MNIST with convolutions](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Answer.ipynb)


