## Week 1 A New Programming Paradigm

#### I. A primer in machine learning

![W1.1](https://raw.githubusercontent.com/JiaRuiShao/TensorFlow/master/1-Introduction%20to%20Tensorflow%20for%20AI%2C%20ML%20and%20DL/images/W1.1.PNG?raw=true "W1.1")

+ Rules and data go in answers come out. Rules are expressed in a programming language and data can come from a variety of sources from local variables all the way up to databases. 
+ Machine learning rearranges this diagram where we put answers in data in and then we get rules out.

**Why we can benefit from this?**

- for some problems, we can't solve them by figuring the rules out for ourselves
- have the computers figure out the rules so we don't have to do it by ourselves

Example:

__activity recognition__

![W1.2](https://github.com/JiaRuiShao/TensorFlow/blob/master/1-Introduction%20to%20Tensorflow%20for%20AI,%20ML%20and%20DL/images/W1.2%20activity%20recognition.PNG?raw=true "W1.2")

__Labeling the data__
The new paradigm is that I get lots and lots of examples and then I have labels on those examples and I use the data to say this is what walking looks like, this is what running looks like, this is what biking looks like and yes, even this is what golfing looks like. So, then it becomes answers and data in with rules being inferred by the machine.

![W1.3](https://raw.githubusercontent.com/JiaRuiShao/TensorFlow/master/1-Introduction%20to%20Tensorflow%20for%20AI%2C%20ML%20and%20DL/images/W1.3.PNG)

A machine learning algorithm then __figures out the specific patterns in each set of data that determines the distinctiveness of each__. That's what's so powerful and exciting about this programming paradigm. It's more than just a new way of doing the same old thing. It opens up new possibilities that were infeasible to do before. 

#### II. An example -- fitting numbers to a line

Here's our first line of code. This is written using Python and TensorFlow and an API in TensorFlow called keras. Keras makes it really easy to define neural networks. 

A neural network is basically a set of functions which can learn patterns.

```python
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
```

The simplest possible neural network is one that has only one neuron in it, and that's what this line of code does. In keras, you use the word __dense__ to define __a layer of connected neurons__. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron. 

__Successive layers__ are defined in sequence, hence the word __sequential__.

You define the shape of what's input to the neural network in the first and in this case the only layer, and you can see that our input shape is super simple. It's just one value. 

You've probably seen that for machine learning, you need to know and use a lot of math, calculus probability and the like. It's really good to understand that as you want to optimize your models but the nice thing for now about TensorFlow and keras is that __a lot of that math is implemented for you in functions__.

**loss functions and optimizers**:

```python
model=compile(optimizer='sgd',loss='mean_squared_error')
 ```

In this case, the loss is mean squared error and the optimizer is SGD which stands for _stochastic gradient descent_. If you want to learn more about these particular functions, as well as the other options that might be better in other scenarios, check out the TensorFlow documentation. 

Our next step is to represent the known data. These are the Xs and the Ys that you saw earlier. The np.array is using a Python library called numpy that makes data representation particularly enlists much easier. 

```python
xs = np.array([-1,0,1,2,3,4],dtype=float)
ys = np.array([-3,-1,1,3,5,7],dtype=float)
 ```

The training takes place in the fit command. Here we're asking the model to figure out how to fit the X values to the Y values. 

```python
model.fit(xs,ys,epochs=500)
 ```

The epochs equals 500 value means that it will go through the training loop 500 times. This training loop is what we described earlier. Make a guess, measure how good or how bad the guesses with the loss function, then use the optimizer and the data to make another guess and repeat this. When the model has finished training, it will then give you back values using the predict method.

```python
print(model.predict([10.0]))
 ```

Now you might think it would return 19 because after all Y equals 2X minus 1, and you think it should be 19. But when you try this in the workbook yourself, you'll see that it will return a value very close to 19 but not exactly 19. Now why do you think that would be? Ultimately there are two main reasons. 

The first is that you trained it using very little data. There's only six points. Those six points are linear but there's no guarantee that for every X, the relationship will be Y equals 2X minus 1. There's a very high probability that Y equals 19 for X equals 10, but the neural network isn't positive. So it will figure out a realistic value for Y. 

The second main reason is when using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit. Keep that in mind as you work through the code. 


##### Summary

So far we've given an __introduction to the concepts and paradigms of Machine Learning and Deep Learning__. You saw that the traditional paradigm of expressing rules in a coding language may not always work to solve a problem. As such, scenarios such as Computer Vision are very difficult to solve with rules-based programming. Instead, if we feed a computer with enough data that we describe (or label) as what we want it to recognize, given that computers are really good at processing data and finding patterns that match, then we could potentially ‘train’ a system to solve a problem.

#### II. Hands-on Exercise

some ways to code:

-  [Jupyter Notebooks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb  "Google Colaboratory in the browser")
- [Google CodLab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)

Now while this might seem very simple, you’ve actually gotten the basics for how neural networks work. As your applications get more complex, you’ll continue to use the same techniques. 

#### III. Quiz

1. The diagram for traditional programming had Rules and Data In, but what came out?
	
	Answers

2. The diagram for Machine Learning had Answers and Data In, but what came out?
	
	Rules

3. When I tell a computer what the data represents (i.e. this data is for walking, this data is for running), what is that process called?
	
	Labelling the data

4. What is a Dense?
	
	A layer of connected neurons

5. What does a Loss function do?
	
	Measures how good the current ‘guess’ is

6. What does the optimizer do?
	
	Generates a new and improved guess

7. What is Convergence?

	The process of getting very close to the correct 		answer

8. What does model.fit do?

	[x] It trains the neural network to fit one set of values to another

	[ ] It optimizes an existing model

	[ ] It determines if your activity is good for your body

	[ ] It makes a model fit available memory


## Weekly Exercise -- Your first Neural Network

#### Get started with Google Colaboratory

colab - jupyter notebook stored in Google Drive

Features:

- Dynamic-rich output (through third-party build-in library)
- Mathematical formula
- Collection of Interactive ML Examples -- [SeedBank](https://research.google.com/seedbank/)

##### Exercise 1 (Housing Prices)

**Question**
Earlier this week you saw a ‘Hello World’ in Machine Learning that predicted a relationship between X and Y values. These were purely arbitrary, but it did give you the template for how you can solve more difficult problems. So, for this exercise you will write code that does a similar task -- in this case predicting house prices based on a simple, linear equation.

[Question](https://colab.research.google.com/drive/19tj2SWfGeAUzNHxt7zz12tLReRqKz6Ae)

[Answer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%201%20-%20House%20Prices/Exercise_1_House_Prices_Answer.ipynb#scrollTo=PUNO2E6SeURH)


## Additional Resources

AI For Everyone is a non-technical course that will help you understand many of the AI technologies we will discuss later in this course, and help you spot opportunities in applying this technology to solve your problems. 
[AI for aeveryone](https://www.deeplearning.ai/ai-for-everyone/)

TensorFlow is available [here](https://www.tensorflow.org/), and video updates from the TensorFlow team are at [Youtube TensorFlow](youtube.com/tensorflow)

Play with a neural network right at [playground tensorflow](http://playground.tensorflow.org/). See if you can figure out the parameters to get the neural network to pattern match to the desired groups. The spiral is particularly challenging!




