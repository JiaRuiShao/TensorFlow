
# Beyond Hello World, A Computer Vision Example
In the previous exercise you saw how to create a neural network that figured out the problem you were trying to solve. This gave an explicit example of learned behavior. Of course, in that instance, it was a bit of overkill because it would have been easier to write the function Y=2x-1 directly, instead of bothering with using Machine Learning to learn the relationship between X and Y for a fixed set of values, and extending that for all values.

But what about a scenario where writing rules like that is much more difficult -- for example a computer vision problem? Let's take a look at a scenario where we can recognize different items of clothing, trained from a dataset containing 10 different types.

## Start Coding

Let's start with our import of TensorFlow


```python
import tensorflow as tf
print(tf.__version__)
```

    C:\Users\surface\ananew\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    

    1.13.1
    

The Fashion MNIST data is available directly in the tf.keras datasets API. You load it like this:


```python
mnist = tf.keras.datasets.fashion_mnist
```

Calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.



```python
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 2s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 0s 0us/step
    

What does these values look like? Let's print a training image, and a training label to see...Experiment with different indices in the array. For example, also take a look at index 42...that's a a different boot than the one at index 0



```python
training_images.shape[0]
```




    60000




```python
training_images.shape[1] # why is not 28*28??
```




    28




```python
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
```

    9
    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
        0   1   4   0   0   0   0   1   1   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
       54   0   0   0   1   3   4   0   0   3]
     [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
      144 123  23   0   0   0   0  12  10   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
      107 156 161 109  64  23  77 130  72  15]
     [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
      216 163 127 121 122 146 141  88 172  66]
     [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
      223 223 215 213 164 127 123 196 229   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
      235 227 224 222 224 221 223 245 173   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
      180 212 210 211 213 223 220 243 202   0]
     [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
      169 227 208 218 224 212 226 197 209  52]
     [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
      198 221 215 213 222 220 245 119 167  56]
     [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
      232 213 218 223 234 217 217 209  92   0]
     [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
      222 221 216 223 229 215 218 255  77   0]
     [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
      211 218 224 223 219 215 224 244 159   0]
     [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
      224 234 176 188 250 248 233 238 215   0]
     [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
      255 255 221 234 221 211 220 232 246   0]
     [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
      188 154 191 210 204 209 222 228 225   0]
     [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
      168 219 221 215 217 223 223 224 229  29]
     [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
      239 223 218 212 209 222 220 221 230  67]
     [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
      199 206 186 181 177 172 181 205 206 115]
     [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
      195 191 198 192 176 156 167 177 210  92]
     [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
      210 210 211 188 188 194 192 216 170   0]
     [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
      182 182 181 176 166 168  99  58   0   0]
     [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]
    

You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, a process called '**normalizing**'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:


```python
training_images  = training_images / 255.0
test_images = test_images / 255.0
```

Now you might be wondering why there are 2 sets...training and testing -- remember we spoke about this in the intro? The idea is to have 1 set of data for training, and then another set of data...that the model hasn't yet seen...to see how good it would be at classifying values. After all, when you're done, you're going to want to try it out with data that it hadn't previously seen!

Let's now design the model. There's quite a few new concepts here, but don't worry, you'll get the hang of them. 


```python
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons

Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now. 

**Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!


The next thing to do, now the model is defined, is to actually build it. You do this by compiling it with an optimizer and loss function as before -- and then you train it by calling **model.fit ** asking it to fit your training data to your training labels -- i.e. have it figure out the relationship between the training data and its actual labels, so in future if you have data that looks like the training data, then it can make a prediction for what that data would look like. 


```python
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=100)
```

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    Epoch 1/100
    60000/60000 [==============================] - 6s 97us/sample - loss: 0.4987 - acc: 0.8243
    Epoch 2/100
    60000/60000 [==============================] - 5s 82us/sample - loss: 0.3728 - acc: 0.8649
    Epoch 3/100
    60000/60000 [==============================] - 5s 83us/sample - loss: 0.3358 - acc: 0.8770
    Epoch 4/100
    60000/60000 [==============================] - 6s 97us/sample - loss: 0.3144 - acc: 0.8849
    Epoch 5/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 0.2950 - acc: 0.8895
    Epoch 6/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.2795 - acc: 0.89580s - loss: 0.2799 - acc: 0
    Epoch 7/100
    60000/60000 [==============================] - 7s 122us/sample - loss: 0.2688 - acc: 0.8993
    Epoch 8/100
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.2574 - acc: 0.9033
    Epoch 9/100
    60000/60000 [==============================] - 6s 97us/sample - loss: 0.2478 - acc: 0.9083
    Epoch 10/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 0.2401 - acc: 0.9112
    Epoch 11/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 0.2327 - acc: 0.9131
    Epoch 12/100
    60000/60000 [==============================] - 6s 95us/sample - loss: 0.2237 - acc: 0.9152
    Epoch 13/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.2166 - acc: 0.9183
    Epoch 14/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.2100 - acc: 0.9215
    Epoch 15/100
    60000/60000 [==============================] - 6s 99us/sample - loss: 0.2063 - acc: 0.9224
    Epoch 16/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 0.1997 - acc: 0.9253
    Epoch 17/100
    60000/60000 [==============================] - 6s 103us/sample - loss: 0.1939 - acc: 0.9267
    Epoch 18/100
    60000/60000 [==============================] - 6s 99us/sample - loss: 0.1891 - acc: 0.9287
    Epoch 19/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.1840 - acc: 0.9306
    Epoch 20/100
    60000/60000 [==============================] - 6s 101us/sample - loss: 0.1790 - acc: 0.9330
    Epoch 21/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.1736 - acc: 0.9344
    Epoch 22/100
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.1692 - acc: 0.9368
    Epoch 23/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 0.1676 - acc: 0.9365
    Epoch 24/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 0.1616 - acc: 0.9383
    Epoch 25/100
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.1592 - acc: 0.9398
    Epoch 26/100
    60000/60000 [==============================] - 6s 102us/sample - loss: 0.1541 - acc: 0.9415
    Epoch 27/100
    60000/60000 [==============================] - 6s 100us/sample - loss: 0.1522 - acc: 0.9421
    Epoch 28/100
    60000/60000 [==============================] - 6s 105us/sample - loss: 0.1479 - acc: 0.9444
    Epoch 29/100
    60000/60000 [==============================] - 6s 108us/sample - loss: 0.1444 - acc: 0.9455
    Epoch 30/100
    60000/60000 [==============================] - 6s 105us/sample - loss: 0.1403 - acc: 0.9470
    Epoch 31/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.1375 - acc: 0.9480
    Epoch 32/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 0.1355 - acc: 0.9485
    Epoch 33/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.1330 - acc: 0.9485
    Epoch 34/100
    60000/60000 [==============================] - 6s 95us/sample - loss: 0.1293 - acc: 0.9521
    Epoch 35/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 0.1261 - acc: 0.9517
    Epoch 36/100
    60000/60000 [==============================] - 5s 89us/sample - loss: 0.1235 - acc: 0.9532
    Epoch 37/100
    60000/60000 [==============================] - 6s 104us/sample - loss: 0.1237 - acc: 0.9537
    Epoch 38/100
    60000/60000 [==============================] - 6s 108us/sample - loss: 0.1193 - acc: 0.9555
    Epoch 39/100
    60000/60000 [==============================] - 6s 102us/sample - loss: 0.1169 - acc: 0.9561s
    Epoch 40/100
    60000/60000 [==============================] - 6s 103us/sample - loss: 0.1152 - acc: 0.9568
    Epoch 41/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 0.1126 - acc: 0.9574
    Epoch 42/100
    60000/60000 [==============================] - 6s 100us/sample - loss: 0.1119 - acc: 0.9574
    Epoch 43/100
    60000/60000 [==============================] - 6s 103us/sample - loss: 0.1085 - acc: 0.9597
    Epoch 44/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 0.1090 - acc: 0.9591
    Epoch 45/100
    60000/60000 [==============================] - 6s 106us/sample - loss: 0.1047 - acc: 0.9607
    Epoch 46/100
    60000/60000 [==============================] - 7s 109us/sample - loss: 0.1050 - acc: 0.9593
    Epoch 47/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 0.1040 - acc: 0.9611
    Epoch 48/100
    60000/60000 [==============================] - 6s 105us/sample - loss: 0.1005 - acc: 0.9622
    Epoch 49/100
    60000/60000 [==============================] - 6s 104us/sample - loss: 0.0992 - acc: 0.9622
    Epoch 50/100
    60000/60000 [==============================] - 8s 139us/sample - loss: 0.0959 - acc: 0.9634
    Epoch 51/100
    60000/60000 [==============================] - ETA: 0s - loss: 0.0979 - acc: 0.963 - 8s 128us/sample - loss: 0.0981 - acc: 0.9631
    Epoch 52/100
    60000/60000 [==============================] - 8s 138us/sample - loss: 0.0933 - acc: 0.9647
    Epoch 53/100
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0936 - acc: 0.9647
    Epoch 54/100
    60000/60000 [==============================] - 8s 138us/sample - loss: 0.0934 - acc: 0.9644
    Epoch 55/100
    60000/60000 [==============================] - 8s 130us/sample - loss: 0.0885 - acc: 0.9666
    Epoch 56/100
    60000/60000 [==============================] - 8s 129us/sample - loss: 0.0882 - acc: 0.9670
    Epoch 57/100
    60000/60000 [==============================] - 10s 161us/sample - loss: 0.0868 - acc: 0.9672
    Epoch 58/100
    60000/60000 [==============================] - 9s 153us/sample - loss: 0.0867 - acc: 0.9674
    Epoch 59/100
    60000/60000 [==============================] - 8s 129us/sample - loss: 0.0841 - acc: 0.9688
    Epoch 60/100
    60000/60000 [==============================] - 9s 147us/sample - loss: 0.0821 - acc: 0.9690
    Epoch 61/100
    60000/60000 [==============================] - 8s 129us/sample - loss: 0.0820 - acc: 0.9694
    Epoch 62/100
    60000/60000 [==============================] - 10s 164us/sample - loss: 0.0798 - acc: 0.9700
    Epoch 63/100
    60000/60000 [==============================] - 8s 137us/sample - loss: 0.0785 - acc: 0.9711
    Epoch 64/100
    60000/60000 [==============================] - 8s 135us/sample - loss: 0.0779 - acc: 0.9705
    Epoch 65/100
    60000/60000 [==============================] - 9s 144us/sample - loss: 0.0808 - acc: 0.9694
    Epoch 66/100
    60000/60000 [==============================] - 11s 181us/sample - loss: 0.0745 - acc: 0.9719
    Epoch 67/100
    60000/60000 [==============================] - 9s 142us/sample - loss: 0.0749 - acc: 0.9720
    Epoch 68/100
    60000/60000 [==============================] - 9s 149us/sample - loss: 0.0733 - acc: 0.9727s - loss: 0.0
    Epoch 69/100
    60000/60000 [==============================] - 10s 169us/sample - loss: 0.0739 - acc: 0.9722
    Epoch 70/100
    60000/60000 [==============================] - 10s 168us/sample - loss: 0.0723 - acc: 0.9724
    Epoch 71/100
    60000/60000 [==============================] - 8s 125us/sample - loss: 0.0731 - acc: 0.9732
    Epoch 72/100
    60000/60000 [==============================] - 6s 107us/sample - loss: 0.0676 - acc: 0.9744
    Epoch 73/100
    60000/60000 [==============================] - 7s 118us/sample - loss: 0.0699 - acc: 0.9736
    Epoch 74/100
    60000/60000 [==============================] - 6s 98us/sample - loss: 0.0711 - acc: 0.9732
    Epoch 75/100
    60000/60000 [==============================] - 6s 102us/sample - loss: 0.0665 - acc: 0.9758
    Epoch 76/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 0.0667 - acc: 0.97550s - loss: 0
    Epoch 77/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.0683 - acc: 0.97441s - lo
    Epoch 78/100
    60000/60000 [==============================] - 6s 107us/sample - loss: 0.0652 - acc: 0.9758
    Epoch 79/100
    60000/60000 [==============================] - 7s 109us/sample - loss: 0.0658 - acc: 0.9758s - loss: 0.0660 
    Epoch 80/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.0639 - acc: 0.9761
    Epoch 81/100
    60000/60000 [==============================] - 6s 101us/sample - loss: 0.0632 - acc: 0.9766
    Epoch 82/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 0.0637 - acc: 0.9767
    Epoch 83/100
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.0615 - acc: 0.9772
    Epoch 84/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 0.0641 - acc: 0.9761
    Epoch 85/100
    60000/60000 [==============================] - 6s 103us/sample - loss: 0.0594 - acc: 0.9774
    Epoch 86/100
    60000/60000 [==============================] - 7s 109us/sample - loss: 0.0604 - acc: 0.9771s - loss: 0.0604 - acc: 0.977
    Epoch 87/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.0572 - acc: 0.9785
    Epoch 88/100
    60000/60000 [==============================] - 6s 103us/sample - loss: 0.0591 - acc: 0.9780
    Epoch 89/100
    60000/60000 [==============================] - 6s 104us/sample - loss: 0.0616 - acc: 0.9779
    Epoch 90/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 0.0592 - acc: 0.9778
    Epoch 91/100
    60000/60000 [==============================] - 7s 108us/sample - loss: 0.0559 - acc: 0.9791
    Epoch 92/100
    60000/60000 [==============================] - 6s 99us/sample - loss: 0.0565 - acc: 0.9788
    Epoch 93/100
    60000/60000 [==============================] - 6s 101us/sample - loss: 0.0549 - acc: 0.9797
    Epoch 94/100
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0581 - acc: 0.9787
    Epoch 95/100
    60000/60000 [==============================] - 6s 99us/sample - loss: 0.0554 - acc: 0.9796
    Epoch 96/100
    60000/60000 [==============================] - 6s 105us/sample - loss: 0.0516 - acc: 0.9807
    Epoch 97/100
    60000/60000 [==============================] - 6s 101us/sample - loss: 0.0574 - acc: 0.9791
    Epoch 98/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 0.0527 - acc: 0.9800
    Epoch 99/100
    60000/60000 [==============================] - 6s 105us/sample - loss: 0.0534 - acc: 0.9802
    Epoch 100/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.0551 - acc: 0.9803
    




    <tensorflow.python.keras.callbacks.History at 0x146ddbe9860>



Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.9098. This tells you that your neural network is about 91% accurate in classifying the training data. I.E., it figured out a pattern match between the image and the labels that worked 91% of the time. Not great, but not bad considering it was only trained for 5 epochs and done quite quickly.

But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. Let's give it a try:


```python
model.evaluate(test_images, test_labels)
```

    10000/10000 [==============================] - 1s 50us/sample - loss: 0.6732 - acc: 0.8889
    




    [0.6732343255124986, 0.8889]



For me, that returned a accuracy of about .8889, which means it was about 88% accurate. As expected it probably would not do as well with *unseen* data as it did with data it was trained on!  As you go through this course, you'll look at ways to improve this. 

To explore further, try the below exercises:


# Exploration Exercises

### Exercise 1:

For this first exercise run the below code: It creates a set of classifications for each of the test images, and then prints the first entry in the classifications. The output, after you run it is a list of numbers. Why do you think this is, and what do those numbers represent? 


```python
classifications = model.predict(test_images)

print(classifications[0]) # the prediction of the first example in testing dataset
```

    [6.8797893e-24 0.0000000e+00 1.1207443e-33 0.0000000e+00 8.5451968e-32
     6.2038070e-18 3.3223834e-32 1.0266690e-07 6.8054532e-23 9.9999988e-01]
    

<font color = 'green'>
Maybe the nums above represent the mean predicted classified numbers of the pictures in each classification
<font color = 'red'>
wrong

Hint: try running print(test_labels[0]) -- and you'll get a 9. Does that help you understand why this list looks the way it does? 


```python
print(test_labels[0])
```

    9
    

<font color = 'green'>
There're 10 labels in this example in total

### What does this list represent?

1.   It's 10 random meaningless values
2.   It's the first 10 classifications that the computer made
<font color = 'red'>
3.   It's the probability that this item is each of the 10 classes

#### Answer: 

The correct answer is (3)

The output of the model is a list of 10 numbers. These numbers are a probability that the value being classified is the corresponding value, i.e. the first value in the list is the probability that the handwriting is of a '0', the next is a '1' etc. Notice that they are all VERY LOW probabilities.

For the 7, the probability was .999+, i.e. the neural network is telling us that it's almost certainly a 7.

### How do you know that this list tells you that the item is an ankle boot?


1.   There's not enough information to answer that question
<font color = 'red'>
2.   The probability of the 10th element on the list is the biggest, and the ankle boot is labelled 9
<font color = 'black'>
2.   The ankle boot is label 9, and there are 0->9 elements in the list




#### Answer

The correct answer is (2). 

Both the list and the labels are 0 based, so the ankle boot having label 9 means that it is the 10th of the 10 classes. The list having the 10th element being the highest value means that the Neural Network has predicted that the item it is classifying is most likely an ankle boot

## Exercise 2: 

Let's now look at the layers in your model. Experiment with different values for the dense layer with 512 neurons. 

What different results do you get for loss, training time etc? Why do you think that's the case? 


```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 1s 0us/step
    Epoch 1/5
    60000/60000 [==============================] - 23s 376us/sample - loss: 0.2026
    Epoch 2/5
    60000/60000 [==============================] - 23s 383us/sample - loss: 0.0808
    Epoch 3/5
    60000/60000 [==============================] - 25s 421us/sample - loss: 0.0527
    Epoch 4/5
    60000/60000 [==============================] - 24s 392us/sample - loss: 0.0360
    Epoch 5/5
    60000/60000 [==============================] - 23s 390us/sample - loss: 0.0271
    10000/10000 [==============================] - 1s 112us/sample - loss: 0.0598
    [8.5191736e-11 1.0198766e-09 2.4212254e-08 3.7656766e-05 1.0017182e-12
     2.1229322e-08 5.9425589e-14 9.9996233e-01 5.6192588e-09 2.4714769e-08]
    7
    

<font color = 'green'>
We can see from the processing time(which is significantly longer) and the accuracy of testing dataset(which improves a lot -- 0.0598 loss compared to 0.6732 loss) that deep neural network with more units works bettwe than shallow neural network with fewer units. That is to say, the neural network with more hidden layers and hidden units is better than the ones with fewer hidden layers and hidden units.

Also, the classification prediction of the first example in testing dataset is different from the previous neural network model 

### Question 1. Increase to 1024 Neurons -- What's the impact?
<font color = 'red'>
1. Training takes longer, but is more accurate
<font color = 'black'>
2. Training takes longer, but no impact on accuracy
3. Training takes the same time, but is more accurate


#### Answer

The correct answer is (1) by adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact -- we do get more accurate. That doesn't mean it's always a case of 'more is better', you can hit the law of diminishing returns very quickly!

## Exercise 3: 

What would happen if you remove the Flatten() layer. Why do you think that's the case? 

You get an error about the shape of the data. It may seem vague right now, but it **reinforces the rule of thumb** that **the first layer in your network should be the same shape as your data**. Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible, so it makes more sense to 'flatten' that 28,28 into a 784x1. 
<font color = 'green'>
Instead of wriitng all the code to handle that ourselves, we add the Flatten() layer at the begining, and when the arrays are loaded into the model later, they'll automatically be flattened for us.


```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([#tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Epoch 1/5
    


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-15-cd0411f1d295> in <module>()
         16               loss = 'sparse_categorical_crossentropy')
         17 
    ---> 18 model.fit(training_images, training_labels, epochs=5)
         19 
         20 model.evaluate(test_images, test_labels)
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, max_queue_size, workers, use_multiprocessing, **kwargs)
        878           initial_epoch=initial_epoch,
        879           steps_per_epoch=steps_per_epoch,
    --> 880           validation_steps=validation_steps)
        881 
        882   def evaluate(self,
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\engine\training_arrays.py in model_iteration(model, inputs, targets, sample_weights, batch_size, epochs, verbose, callbacks, val_inputs, val_targets, val_sample_weights, shuffle, initial_epoch, steps_per_epoch, validation_steps, mode, validation_in_fit, **kwargs)
        327 
        328         # Get outputs.
    --> 329         batch_outs = f(ins_batch)
        330         if not isinstance(batch_outs, list):
        331           batch_outs = [batch_outs]
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\backend.py in __call__(self, inputs)
       3074 
       3075     fetched = self._callable_fn(*array_vals,
    -> 3076                                 run_metadata=self.run_metadata)
       3077     self._call_fetch_callbacks(fetched[-len(self._fetches):])
       3078     return nest.pack_sequence_as(self._outputs_structure,
    

    ~\ananew\lib\site-packages\tensorflow\python\client\session.py in __call__(self, *args, **kwargs)
       1437           ret = tf_session.TF_SessionRunCallable(
       1438               self._session._session, self._handle, args, status,
    -> 1439               run_metadata_ptr)
       1440         if run_metadata:
       1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    ~\ananew\lib\site-packages\tensorflow\python\framework\errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
        526             None, None,
        527             compat.as_text(c_api.TF_Message(self.status.status)),
    --> 528             c_api.TF_GetCode(self.status.status))
        529     # Delete the underlying status object from memory otherwise it stays alive
        530     # as there is a reference to status from this from the traceback due to
    

    InvalidArgumentError: logits and labels must have the same first dimension, got logits shape [896,10] and labels shape [32]
    	 [[{{node loss_2/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits}}]]


## Exercise 4: 

Consider the final (output) layers. Why are there 10 of them? What would happen if you had a different amount than 10? For example, try training the network with 5

You get an error as soon as it finds an unexpected value. **Another rule of thumb**-- **the number of neurons in the last layer should match the number of classes you are classifying for**. In this case it's the digits 0-9, so there are 10 of them, hence you should have 10 neurons in your final layer.


```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Epoch 1/5
    


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-17-04e5ea87fa1d> in <module>()
         16               loss = 'sparse_categorical_crossentropy')
         17 
    ---> 18 model.fit(training_images, training_labels, epochs=5)
         19 
         20 model.evaluate(test_images, test_labels)
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, max_queue_size, workers, use_multiprocessing, **kwargs)
        878           initial_epoch=initial_epoch,
        879           steps_per_epoch=steps_per_epoch,
    --> 880           validation_steps=validation_steps)
        881 
        882   def evaluate(self,
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\engine\training_arrays.py in model_iteration(model, inputs, targets, sample_weights, batch_size, epochs, verbose, callbacks, val_inputs, val_targets, val_sample_weights, shuffle, initial_epoch, steps_per_epoch, validation_steps, mode, validation_in_fit, **kwargs)
        327 
        328         # Get outputs.
    --> 329         batch_outs = f(ins_batch)
        330         if not isinstance(batch_outs, list):
        331           batch_outs = [batch_outs]
    

    ~\ananew\lib\site-packages\tensorflow\python\keras\backend.py in __call__(self, inputs)
       3074 
       3075     fetched = self._callable_fn(*array_vals,
    -> 3076                                 run_metadata=self.run_metadata)
       3077     self._call_fetch_callbacks(fetched[-len(self._fetches):])
       3078     return nest.pack_sequence_as(self._outputs_structure,
    

    ~\ananew\lib\site-packages\tensorflow\python\client\session.py in __call__(self, *args, **kwargs)
       1437           ret = tf_session.TF_SessionRunCallable(
       1438               self._session._session, self._handle, args, status,
    -> 1439               run_metadata_ptr)
       1440         if run_metadata:
       1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    ~\ananew\lib\site-packages\tensorflow\python\framework\errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
        526             None, None,
        527             compat.as_text(c_api.TF_Message(self.status.status)),
    --> 528             c_api.TF_GetCode(self.status.status))
        529     # Delete the underlying status object from memory otherwise it stays alive
        530     # as there is a reference to status from this from the traceback due to
    

    InvalidArgumentError: Received a label value of 9 which is outside the valid range of [0, 5).  Label values: 2 8 1 7 1 4 2 9 0 5 5 3 7 5 8 4 5 1 4 5 6 8 2 9 8 9 5 9 0 5 1 2
    	 [[{{node loss_4/output_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits}}]]


## Exercise 5: 

Consider the effects of the amount of units in hidden layers in the network. What will happen if you add another layer between the one with 512 and the final layer with 10. 

Ans: There isn't a significant impact -- because this is relatively simple data. For far more complex data (including color images to be classified as flowers that you'll see in the next lesson), extra layers are often necessary. 


```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Epoch 1/5
    60000/60000 [==============================] - 29s 476us/sample - loss: 0.1868
    Epoch 2/5
    60000/60000 [==============================] - 28s 469us/sample - loss: 0.0801
    Epoch 3/5
    60000/60000 [==============================] - 29s 483us/sample - loss: 0.0535
    Epoch 4/5
    60000/60000 [==============================] - 27s 454us/sample - loss: 0.0419
    Epoch 5/5
    60000/60000 [==============================] - 27s 456us/sample - loss: 0.0331
    10000/10000 [==============================] - 1s 141us/sample - loss: 0.0727
    [4.5655229e-12 1.1329102e-08 1.3262921e-10 1.9695817e-07 2.3211833e-11
     3.3863305e-12 1.6943873e-14 9.9999976e-01 7.5870539e-11 6.9224488e-09]
    7
    

## Exercise 6: 

Consider the impact of training for more or less epochs. Why do you think that would be the case? 

Try 15 epochs -- you'll probably get a model with a worse loss of testing dataset than the one with 5 epochs -- **you might see the loss value stops decreasing, and sometimes increases. This is a side effect of something called 'overfitting' which you can learn about [somewhere] and it's something you need to keep an eye out for when training neural networks**. There's no point in wasting your time training if you aren't improving your loss, right! :)


```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=15)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[34])
print(test_labels[34])
```

    1.13.1
    Epoch 1/15
    60000/60000 [==============================] - 24s 396us/sample - loss: 0.1990
    Epoch 2/15
    60000/60000 [==============================] - 24s 404us/sample - loss: 0.0804
    Epoch 3/15
    60000/60000 [==============================] - 25s 414us/sample - loss: 0.0523
    Epoch 4/15
    60000/60000 [==============================] - 26s 431us/sample - loss: 0.0374
    Epoch 5/15
    60000/60000 [==============================] - 26s 432us/sample - loss: 0.0272
    Epoch 6/15
    60000/60000 [==============================] - 25s 420us/sample - loss: 0.0212
    Epoch 7/15
    60000/60000 [==============================] - 25s 416us/sample - loss: 0.0175
    Epoch 8/15
    60000/60000 [==============================] - 25s 412us/sample - loss: 0.0139
    Epoch 9/15
    60000/60000 [==============================] - 24s 395us/sample - loss: 0.0132
    Epoch 10/15
    60000/60000 [==============================] - 26s 426us/sample - loss: 0.0118
    Epoch 11/15
    60000/60000 [==============================] - 25s 417us/sample - loss: 0.0098
    Epoch 12/15
    60000/60000 [==============================] - 24s 395us/sample - loss: 0.0095
    Epoch 13/15
    60000/60000 [==============================] - 26s 432us/sample - loss: 0.0093
    Epoch 14/15
    60000/60000 [==============================] - 26s 425us/sample - loss: 0.0085
    Epoch 15/15
    60000/60000 [==============================] - 26s 430us/sample - loss: 0.0088
    10000/10000 [==============================] - 1s 132us/sample - loss: 0.0975
    [1.2530482e-21 8.5301470e-18 5.5552146e-11 2.4379329e-13 8.7311718e-32
     2.6624332e-30 8.2008254e-26 1.0000000e+00 3.5901872e-18 3.3067760e-20]
    7
    

<font color = 'green'>
The loss of the testing data when epochs equals to 5 is about 0.0598

## Exercise 7: 

Before you trained, you normalized the data, going from values that were 0-255 to values that were 0-1. What would be the impact of removing that? Here's the complete code to give it a try. Why do you think you get different results? 

<font color = 'green'>
train the model with unnormalized data


```python
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Epoch 1/5
    60000/60000 [==============================] - 7s 120us/sample - loss: 9.2230
    Epoch 2/5
    60000/60000 [==============================] - 7s 121us/sample - loss: 7.8654
    Epoch 3/5
    60000/60000 [==============================] - 9s 155us/sample - loss: 7.6720
    Epoch 4/5
    60000/60000 [==============================] - 9s 152us/sample - loss: 7.5210
    Epoch 5/5
    60000/60000 [==============================] - 7s 109us/sample - loss: 7.4646s - loss: 7
    10000/10000 [==============================] - 1s 67us/sample - loss: 7.2054
    [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    7
    

<font color = 'green'>
train the model with unnormalized data


```python
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
```

    1.13.1
    Epoch 1/5
    60000/60000 [==============================] - 8s 127us/sample - loss: 0.2596s - loss: 0.2 - ETA: 0s - los
    Epoch 2/5
    60000/60000 [==============================] - 7s 115us/sample - loss: 0.1128
    Epoch 3/5
    60000/60000 [==============================] - 6s 107us/sample - loss: 0.0774
    Epoch 4/5
    60000/60000 [==============================] - 7s 118us/sample - loss: 0.0580
    Epoch 5/5
    60000/60000 [==============================] - 7s 118us/sample - loss: 0.0453
    10000/10000 [==============================] - 1s 69us/sample - loss: 0.0682
    [2.3136521e-07 4.7205134e-08 6.7421046e-05 1.5692001e-04 1.7808109e-10
     2.7456924e-06 1.8238536e-12 9.9968517e-01 2.4499877e-05 6.2899016e-05]
    7
    

## Exercise 8: 

Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you, and if you reach that after 3 epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other program...you have callbacks! Let's see them in action...


```python
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

```
