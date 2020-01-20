## Sequences, Time Series and Prediction

### Week 1 Sequences and Prediction

**Types of Time Series**:

- univariate time series 

- multivariate time series

![multi-variate](https://image.slidesharecdn.com/adersberger-timeseries-analysis-161017103935/95/time-series-analysis-6-638.jpg?cb=1501067462)

Multivariate Time Series charts can be useful ways of understanding the impact of related data.

**Machine learning applied to time series**

- prediction of forecasting based on the data

- imputation \-\-project back into the past to see how we got to where we are now

- to detect anomalies

- to analyze the time series to spot patterns in them that determine what generated the series itself(speech recognition)

**Common patterns in time series**

1. trend

2. seasonality

3. white noise(no trend and no seasonality) \-\- not much we can do (unpredictable!!)

4. auto correlated time series \-\- it correlates with a delayed copy of itself often called a lag

![1_1](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_1.PNG?raw=true)

Often a time series like this is described as having memory as steps are dependent on previous ones.

**Time series we'll encounter in real life = trend + seasonality + autocorrelation + noise**

![1_3](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_3.PNG?raw=true)

As we've learned a machine-learning model is designed to spot patterns, and when we spot patterns we can make predictions. 


**Types**

- non-stationary time series

![1_2](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_2.PNG?raw=true)

To predict on this we could just train for limited period of time. For example, here where I take just the last 100 steps. You'll probably get a better performance than if you had trained on the entire time series. But that's breaking the mold for typical machine, learning where we always assume that more data is better

Ideally, we would like to be able to take the whole series into account and generate a prediction for what might happen next

the optimal time window that you should use for training will vary

- stationary time series

its behavior does not change over time

the more data you have, the better


**Introduction to time series**

[Notebook -- Introduction to time series](https://colab.research.google.com/drive/1PxHfGj_0eiHkESPMybfiRvnol9xKyNZH)


**Train, validation and test sets**

Naive Forcasting -- take the last value and assume that the next value will be the same one

![1_5](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_5.PNG?raw=true)


measure performance? 

1. __fixed partitioning__

We typically want to split the time series into a training period, a validation period and a test period. This is called __*fixed partitioning*__. 

Next you'll train your model on the training period, and you'll evaluate it on the validation period. 

Here's where you can experiment to find the right architecture for training. And work on it and your hyper parameters, until you get the desired performance, measured using the validation set. 

Often, once you've done that, you can retrain using both the training and validation data. And then test on the test period to see if your model will perform just as well. 

And if it does, then you could take the unusual step of retraining again, using also the test data. But why would you do that? Well, it's because the test data is the closest data you have to the current point in time. And as such it's often the strongest signal in determining future values. If your model is not trained using that data, too, then it may not be optimal. 

Due to this, it's actually quite common to forgo a test set all together. And just train, using a training period and a validation period, and the test set is in the future.

2. **roll-forward partitioning**

We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period. 

You could see it as doing fixed partitioning a number of times, and then continually refining the model as such.


**Metrics for evaluating performance**

![1_6](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_6.PNG?raw=true)

The most common metric to evaluate the forecasting performance of a model is the mean squared error or mse where we square the errors and then calculate their mean.

**MAE(Mean Absolute Error)** does not penalize large errors as much as the mse does. Depending on your task, you may prefer the mae or the mse. 

For example, if large errors are potentially dangerous and they cost you much more than smaller errors, then you may prefer the mse. But if your gain or your loss is just proportional to the size of the error, then the mae may be better.

Also, you can measure the **MAPE(mean absolute percentage error)** or mape, this is the mean ratio between the absolute error and the absolute value, this gives an idea of the size of the errors compared to the values.

```python
keras.metrics.mean_absolute_error(x_valid, naive_forcast).numpy()

```

The keras metrics libraries include an MAE that can be called like this.

**Moving average and differencing**

A common and very simple forecasting method is to calculate a **moving average**. 

![1_7](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_7.PNG?raw=true)

This nicely eliminates a lot of the noise and it gives us a curve roughly emulating the original series, but it does not anticipate trend or seasonality.

Depending on the current time i.e. the period after which you want to forecast for the future, it can actually end up being worse than a naive forecast.

One method to avoid this is to remove the trend and seasonality from the time series with a technique called **differencing**. 

![1_8](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_8.PNG?raw=true)

Optimal Way to measure: Differencing + Moving Average

![1_9](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_9.PNG?raw=true)

To get the final forecasts for the original time series, we just need to add back the value at time T minus 365, and we'll get these forecasts.

![1_10](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_10.PNG?raw=true)

You may have noticed that our moving average removed a lot of noise but our final forecasts are still pretty noisy. Where does that noise come from? Well, that's coming from the past values that we added back into our forecasts. So we can improve these forecasts by also removing the past noise using a moving average on that.

![1_11](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_11.PNG?raw=true)

Keep this in mind before you rush into deep learning. Simple approaches sometimes can work just fine.


**Forcasting on the synthetic dataset**

Note: Before you start, make sure you are running Python 3 and you're using an environment that provides a GPU. Some of the code will require TensoFlow 2.0 to be installed. So make sure that you have it.

[Notebook -- Forcasting on the synthetic dataset](https://colab.research.google.com/drive/1_usmArauRRcx_ginYuwpb9n_7BKABseQ#scrollTo=iN2MsBxWTE3m)

**Exercise -- Create and predict synthetic data**

[Exercise](https://colab.research.google.com/drive/1iNSFtSJk-M-Zprtx1KCQXrFhDV0q01Rd)


### Week 2 Deep Neural Network for Time Series 

In this week, you'll learn to apply a new network to these sequences. We'll start with a relatively simple DNN, and you'll learn how to tune the learning rate of the optimizer.

**Preparing features and labels**

**imput**: a number of values in the series

**label**: the next value

First of all, as with any other ML problem, we have to divide our data into features and labels. In this case our feature is effectively a number of values in the series, with our label being the next value. We'll call that number of values that will treat as our feature, the window size, where we're taking a window of the data and training an ML model to predict the next value. So for example, if we take our time series data, say, 30 days at a time, we'll use 30 values as the feature and the next value is the label. Then over time, we'll train a neural network to match the 30 features to the single label.

Example: use the `tf.data.Dataset` class to create some data for us, we'll make a range of 10 values. When we print them we'll see a series of data from 0 to 9. To make it more interesting, we'll use the `dataset.window` to expand our data set using windowing.

![1_12](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_12.PNG?raw=true)

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
dataset = dataset.flat
for windoe_dataset in dataset:
	for val in window_dataset:
		print(val.numpy(), end=" ")
	print()

```

![1_13](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_13.PNG?raw=true)


Use `drop_remainder` to drop the non-five chunks:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for windoe_dataset in dataset:
	for val in window_dataset:
		print(val.numpy(), end=" ")
	print()

```

![1_14](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_14.PNG?raw=true)

put these into numpy lists so that we can start using them with machine learning:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
	print(window.numpy())

```

![1_15](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_15.PNG?raw=true)

split the data into features and labels:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
	print(x.numpy(), y.numpy())

```

![1_16](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_16.PNG?raw=true)

shuffle the data before training(so as not to accidentally introduce a sequence bias): 

Sequence bias is when the order of things can impact the selection of things. For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up.

Using a shuffle buffer speeds things up a bit. So for example, if you have 100,000 items in your dataset, but you set the buffer to a thousand. It will just fill the buffer with the first thousand elements, pick one of them at random. And then it will replace that with the 1,000 and first element before randomly picking again, and so on. This way with super large datasets, the random element choosing can choose from a smaller number which effectively speeds things up.

**??? may not be applied in time-series model if sequence order matters**

```python
dataset = tf.data.Dataset.range(10) # create data items
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10) # We call it with the buffer size of ten, because that's the amount of data items that we have
for x,y in dataset:
	print(x.numpy(), y.numpy())

```

![1_17](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_17.PNG?raw=true)

batching the data:

```python
dataset = tf.data.Dataset.range(10) 
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
	print("x = ", x.numpy())
	print("y = ", y.numpy())

```

![1_18](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_18.PNG?raw=true)


[Exercise -- Prepraing the data before training](https://colab.research.google.com/drive/1swlYnl18UGoXZfsgAc4RcPxsN_nrRmYr#scrollTo=Wa0PNwxMGapy)


**Feeding windowed dataset into neural network**

![1_19](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_19.PNG?raw=true)

So if you think back to this diagram and you consider the input window to be 20 values wide, then let's call them x0, x1, x2, etc, all the way up to x19. But let's be clear. That's not the value on the horizontal axis which is commonly called the x-axis, it's the value of the time series at that point on the horizontal axis. So the value at time t0, which is 20 steps before the current value is called x0, and t1 is called x1, etc. Similarly, for the output, which we would then consider to be the value at the current time to be the y.

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series) # create a dataset from the series
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # slice the data up into the appropriate windows. Each one being shifted by one time set. We'll keep them all the same size by setting drop remainder to true
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) #  flatten the data out to make it easier to work with
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1])) # shuffle the data
  dataset = dataset.batch(batch_size).prefetch(1) # batch the data
  return dataset
```

**Single layer neural network -- linear regression**

```python

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size]) # create a single dense layer with its input shape being the window size
model = tf.keras.models.Sequential([l0])

# compile and fit the model
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)) # setting loss to MSE, and here optimizer we use Stochastic Gradient Descent
model.fit(dataset,epochs=100,verbose=0) # Ignoring the epoch by setting verbose to zero

# print the weights
print("Layer weights {}".format(l0.get_weights()))
```

**Prediction**

![1_20](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_20.PNG?raw=true)

```python
forecast = []

for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
```

**Exercise -- Build a one-layer neural network**

[Exercise Notebook](https://colab.research.google.com/drive/18VqCeMd09RGU8sDB207LTJMoF1sI6GN1#scrollTo=hR2BO0Dai_ZT)

**Deep(3-layer) neural network training, tuning and prediction**

```python
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)) # compile the model
model.fit(dataset,epochs=100,verbose=0)
```



```python
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```

pick the optimal learning rate using `callback`:

```python

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

# callback at the end of each epoch
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0) # call callback fuction here

```

pick up the learning rate with the lowest cost:

```python
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```

![1_21](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_21.PNG?raw=true)

update the model using the choosen learning rate:

```python
window_size = 30
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9) # choosen learning rate is roughly 8e-6

model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=500, verbose=0)
```

**Exercise**

[Exercise Notebook -- Build a Simple Deep Neural Network Model](https://colab.research.google.com/drive/1zj_2lZhJTYaG7-YY18KYDG03kXGZ_uKG#scrollTo=QDwW0Q7ovYK1)

[Weekly Exercise -- Predict with a DNN](https://colab.research.google.com/drive/1Ge5DTv9J8r4dcHFtZDRfefSMAQzp4plt#scrollTo=TW-vT7eLYAdb)

[Weekly Exercise -- Predict with a DNN (practice version)](https://colab.research.google.com/drive/1RPhdKvNrWj9GzQ5Vq0aKRS-BeGEeOcZS)

### Week 3 Recurrent Neural Network for Time Series

**Recurrent Neural Network**

![1_22](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_22.PNG?raw=true)

With an RNN, you can feed it in batches of sequences, and it will output a batch of forecasts.

**The first dimension will be the batch size, the second will be the timestamps, and the third is the dimensionality of the inputs at each time step(series dimensionality)**. 

For example, if it's a univariate time series, this value will be one, for multivariate it'll be more. 

**How RNN layers work**

![1_23](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_23.PNG?raw=true)

**Shape of the inputs to the RNN**

![1_24](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_24.PNG?raw=true)

So for example, if we have a window size of 30 timestamps and we're batching them in sizes of four, the shape will be 4 times 30 times 1, and each timestamp, the memory cell input will be a four by one matrix, like this. 

For subsequent ones, it'll be the output from the memory cell. If the memory cell is comprised of three neurons, then the output matrix will be four by three because the batch size coming in was four and the number of neurons is three. So the full output of the layer is three dimensional, in this case, 4 by 30 by 3.

**Outputting a sequence**

![1_25](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_25.PNG?raw=true)

the last unit is just one because we're using a univariate time series

If we set return_sequences to true and all recurrent layers, then they will all output sequences and the dense layer will get a sequence as its inputs. 

![1_26](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_26.PNG?raw=true)

Keras handles this by using the same dense layer independently at each time stamp. It might look like multiple ones here but it's the same one that's being reused at each time step. This gives us what is called a sequence to sequence RNN. It's fed a batch of sequences and it returns a batch of sequences of the same length. The dimensionality may not always match. It depends on the number of units in the memory sale. 

**Lambda layers**

![1_27](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_27.PNG?raw=true)

Lambda Layer: allows us to perform arbitrary operations to effectively expand the functionality of TensorFlow's kares

The first Lambda layer will be used to help us with our dimensionality. With the Lambda layer, we can fix this without rewriting our Window dataset helper function. Using the Lambda, we just expand the array by one dimension. By setting input shape to none, we're saying that the model can take sequences of any length. 

Scaling up the outputs to the same ballpark can help us with learning. We can do that in a Lambda layer too, we just simply multiply that by a 100. 

**Adjusting the learning rate dynamically**

optimize the neural network for the learning rate of the optimizer:

So here's the code for training the RNN with two layers each with 40 cells. To tune the learning rate, we'll set up a callback, which you can see here.

![1_28](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_28.PNG?raw=true)

New loss function: [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)

The Huber function is a loss function that's less sensitive to outliers and as this data can get a little bit noisy, it's worth giving it a shot.

**RNN**

[Exercise -- Build a RNN Model](https://colab.research.google.com/drive/1xarcZRUZ8Q21RQkpAu_MUbKoEegtAef4#scrollTo=Zswl7jRtGzkk)

**Long Short Term Memory Network(LSTM)**

perhaps a better approach would be to use LSTMs instead of RNNs to see the impact

![1_30](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_30.PNG?raw=true)

LSTMs are the cell state to this that keep a state throughout the life of the training so that the state is passed from cell to cell, timestamp to timestamp, and it can be better maintained. This means that the data from earlier in the window can have a greater impact on the overall projection than in the case of RNNs. 

![1_29](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_29.PNG?raw=true)

[Video Lecture](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn)

[Exercise -- Build a LSTM Model](https://colab.research.google.com/drive/1L59VyzxB0sie31SkWvhp8ojtpzTbXZXE#scrollTo=3CGaYFxXNEAK)


**Weekly Exercise - Mean Absolute Error**

In this exercise you’ll take a synthetic data set and write the code to pick the learning rate and then train on it to get an MAE of < 3

[LSTM -- Mean Absolute Error Notebook]()

[LSTM -- Mean Absolute Error Notebook (Exercise)](https://colab.research.google.com/drive/1tC0FA1CndPFKLkBSlDlD57OBUAX-uL77#scrollTo=D1J15Vh_1Jih)


### Week 4 Real-world time series data

**ways to improve the model preformance**:

Experimenting with hyperparameters

- try more epochs(be care of overfitting)

- explore different batch size

- find the optimal window size

- change different values of hidden layer

- find the optimal learning rate

- change the filter size

[more about batch: Mini Batch Gradient Descent](https://www.youtube.com/watch?v=4qJaSmvhxi8)

[LSTM Notebook](https://colab.research.google.com/drive/1LcYjVTKDVW_wyDPexBC1UoYuiF4eeaR2)

**Real data - sunspots**

Data Format:

Index, Date, Monthly Mean Total Sunspot Number

![1_31](https://github.com/JiaRuiShao/TensorFlow/blob/master/4-Sequences,%20Time%20Series%20and%20Prediction/images/1_31.PNG?raw=true)

[LSTM - Sunspot](https://colab.research.google.com/drive/152jU0ruULeMwIulPG-Kc2i4Kd_-XLg3M#scrollTo=AOVzQXxCwkzP)

**Weekly Exercise**

[LSTM - Daily Min Temp](https://colab.research.google.com/drive/1_yhtB-gHh2pBEfO1M2uTWED_3m9ilZf5#scrollTo=GNkzTFfynsmV)


[LSTM - Daily Min Temp (Practice)](https://colab.research.google.com/drive/1pRRXWiz5QQisPd7AU031gysh5D11A11s#scrollTo=sLl52leVp5wU)

