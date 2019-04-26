
# The Hello World of Deep Learning with Neural Networks

Like every first app you should start with something super simple that shows the overall scaffolding for how your code works. 

In the case of creating neural networks, the sample I like to use is one where it learns the relationship between two numbers. So, for example, if you were writing code for a function like this, you already know the 'rules' — 


```
float hw_function(float x){
    float y = (2 * x) - 1;
    return y;
}
```

So how would you train a neural network to do the equivalent task? Using data! By feeding it with a set of Xs, and a set of Ys, it should be able to figure out the relationship between them. 

This is obviously a very different paradigm than what you might be used to, so let's step through it piece by piece.


## Imports

Let's start with our imports. Here we are importing TensorFlow and calling it tf for ease of use.

We then import a library called numpy, which helps us to represent our data as lists easily and quickly.

The framework for defining a neural network as a set of Sequential layers is called keras, so we import that too.


```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
```

    C:\Users\surface\ananew\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    

## Define and Compile the Neural Network

Next we will create the simplest possible neural network. It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.


```python
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

Now we compile our Neural Network. When we do so, we have to specify 2 functions, a loss and an optimizer.

If you've seen lots of math for machine learning, here's where it's usually used, but in this case it's nicely encapsulated in functions for you. But what happens here — let's explain...

We know that in our function, the relationship between the numbers is y=2x-1. 

When the computer is trying to 'learn' that, it makes a guess...maybe y=10x+10. The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.

It then uses the OPTIMIZER function to make another guess. Based on how the loss function went, it will try to minimize the loss. At that point maybe it will come up with somehting like y=5x+5, which, while still pretty bad, is closer to the correct result (i.e. the loss is lower)

It will repeat this for the number of EPOCHS which you will see shortly. But first, here's how we tell it to use 'MEAN SQUARED ERROR' for the loss and 'STOCHASTIC GRADIENT DESCENT' for the optimizer. You don't need to understand the math for these yet, but you can see that they work! :)

Over time you will learn the different and appropriate loss and optimizer functions for different scenarios. 



```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\keras\utils\losses_utils.py:170: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    


```python
tf.cast?
```

## Providing the Data

Next up we'll feed in some data. In this case we are taking 6 xs and 6ys. You can see that the relationship between these is that y=2x-1, so where x = -1, y=-3 etc. etc. 

A python library called 'Numpy' provides lots of array type data structures that are a defacto standard way of doing it. We declare that we want to use these by specifying the values as an np.array[]


```python
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```

# Training the Neural Network

The process of training the neural network, where it 'learns' the relationship between the Xs and Ys is in the **model.fit**  call. This is where it will go through the loop we spoke about above, making a guess, measuring how good or bad it is (aka the loss), using the opimizer to make another guess etc. It will do it for the number of epochs you specify. When you run this code, you'll see the loss on the right hand side.


```python
model.fit(xs, ys, epochs=500)
```

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/500
    6/6 [==============================] - 0s 31ms/sample - loss: 31.9626
    Epoch 2/500
    6/6 [==============================] - 0s 332us/sample - loss: 25.4680
    Epoch 3/500
    6/6 [==============================] - 0s 333us/sample - loss: 20.3518
    Epoch 4/500
    6/6 [==============================] - 0s 332us/sample - loss: 16.3202
    Epoch 5/500
    6/6 [==============================] - 0s 1ms/sample - loss: 13.1420
    Epoch 6/500
    6/6 [==============================] - 0s 830us/sample - loss: 10.6353
    Epoch 7/500
    6/6 [==============================] - 0s 666us/sample - loss: 8.6571
    Epoch 8/500
    6/6 [==============================] - 0s 333us/sample - loss: 7.0947
    Epoch 9/500
    6/6 [==============================] - 0s 998us/sample - loss: 5.8598
    Epoch 10/500
    6/6 [==============================] - 0s 499us/sample - loss: 4.8824
    Epoch 11/500
    6/6 [==============================] - 0s 332us/sample - loss: 4.1079
    Epoch 12/500
    6/6 [==============================] - 0s 664us/sample - loss: 3.4931
    Epoch 13/500
    6/6 [==============================] - 0s 1ms/sample - loss: 3.0040
    Epoch 14/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.6140
    Epoch 15/500
    6/6 [==============================] - 0s 665us/sample - loss: 2.3020
    Epoch 16/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.0514
    Epoch 17/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.8494
    Epoch 18/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.6856
    Epoch 19/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.5520
    Epoch 20/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.4423
    Epoch 21/500
    6/6 [==============================] - 0s 997us/sample - loss: 1.3514
    Epoch 22/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.2754
    Epoch 23/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.2113
    Epoch 24/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.1566
    Epoch 25/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.1094
    Epoch 26/500
    6/6 [==============================] - 0s 664us/sample - loss: 1.0681
    Epoch 27/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.0317
    Epoch 28/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.9990
    Epoch 29/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.9695
    Epoch 30/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.9425
    Epoch 31/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.9176
    Epoch 32/500
    6/6 [==============================] - 0s 998us/sample - loss: 0.8944
    Epoch 33/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.8726
    Epoch 34/500
    6/6 [==============================] - 0s 829us/sample - loss: 0.8519
    Epoch 35/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.8323
    Epoch 36/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.8135
    Epoch 37/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.7955
    Epoch 38/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.7781
    Epoch 39/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.7613
    Epoch 40/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.7450
    Epoch 41/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.7292
    Epoch 42/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.7139
    Epoch 43/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.6989
    Epoch 44/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.6843
    Epoch 45/500
    6/6 [==============================] - 0s 829us/sample - loss: 0.6700
    Epoch 46/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.6561
    Epoch 47/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.6425
    Epoch 48/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.6292
    Epoch 49/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.6162
    Epoch 50/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.6035
    Epoch 51/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.5911
    Epoch 52/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.5789
    Epoch 53/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.5670
    Epoch 54/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.5553
    Epoch 55/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.5439
    Epoch 56/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.5327
    Epoch 57/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.5217
    Epoch 58/500
    6/6 [==============================] - 0s 330us/sample - loss: 0.5110
    Epoch 59/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.5005
    Epoch 60/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.4902
    Epoch 61/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.4802
    Epoch 62/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.4703
    Epoch 63/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.4606
    Epoch 64/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.4512
    Epoch 65/500
    6/6 [==============================] - 0s 165us/sample - loss: 0.4419
    Epoch 66/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.4328
    Epoch 67/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.4239
    Epoch 68/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.4152
    Epoch 69/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.4067
    Epoch 70/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.3983
    Epoch 71/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.3901
    Epoch 72/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.3821
    Epoch 73/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.3743
    Epoch 74/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.3666
    Epoch 75/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.3591
    Epoch 76/500
    6/6 [==============================] - 0s 495us/sample - loss: 0.3517
    Epoch 77/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.3445
    Epoch 78/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.3374
    Epoch 79/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.3305
    Epoch 80/500
    6/6 [==============================] - 0s 994us/sample - loss: 0.3237
    Epoch 81/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.3170
    Epoch 82/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.3105
    Epoch 83/500
    6/6 [==============================] - 0s 3ms/sample - loss: 0.3041
    Epoch 84/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.2979
    Epoch 85/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.2918
    Epoch 86/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.2858
    Epoch 87/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.2799
    Epoch 88/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.2742
    Epoch 89/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.2685
    Epoch 90/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.2630
    Epoch 91/500
    6/6 [==============================] - 0s 830us/sample - loss: 0.2576
    Epoch 92/500
    6/6 [==============================] - 0s 336us/sample - loss: 0.2523
    Epoch 93/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.2471
    Epoch 94/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.2421
    Epoch 95/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.2371
    Epoch 96/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.2322
    Epoch 97/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.2274
    Epoch 98/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.2228
    Epoch 99/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.2182
    Epoch 100/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.2137
    Epoch 101/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.2093
    Epoch 102/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.2050
    Epoch 103/500
    6/6 [==============================] - 0s 335us/sample - loss: 0.2008
    Epoch 104/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.1967
    Epoch 105/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1926
    Epoch 106/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.1887
    Epoch 107/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.1848
    Epoch 108/500
    6/6 [==============================] - 0s 669us/sample - loss: 0.1810
    Epoch 109/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1773
    Epoch 110/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.1737
    Epoch 111/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.1701
    Epoch 112/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1666
    Epoch 113/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1632
    Epoch 114/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1598
    Epoch 115/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.1565
    Epoch 116/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1533
    Epoch 117/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1502
    Epoch 118/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.1471
    Epoch 119/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.1441
    Epoch 120/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1411
    Epoch 121/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1382
    Epoch 122/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1354
    Epoch 123/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1326
    Epoch 124/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.1299
    Epoch 125/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1272
    Epoch 126/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.1246
    Epoch 127/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1220
    Epoch 128/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1195
    Epoch 129/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1171
    Epoch 130/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1147
    Epoch 131/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.1123
    Epoch 132/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1100
    Epoch 133/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.1077
    Epoch 134/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1055
    Epoch 135/500
    6/6 [==============================] - 0s 164us/sample - loss: 0.1034
    Epoch 136/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.1012
    Epoch 137/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0992
    Epoch 138/500
    6/6 [==============================] - 0s 330us/sample - loss: 0.0971
    Epoch 139/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0951
    Epoch 140/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0932
    Epoch 141/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0913
    Epoch 142/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0894
    Epoch 143/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0875
    Epoch 144/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0858
    Epoch 145/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0840
    Epoch 146/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0823
    Epoch 147/500
    6/6 [==============================] - 0s 832us/sample - loss: 0.0806
    Epoch 148/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0789
    Epoch 149/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0773
    Epoch 150/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0757
    Epoch 151/500
    6/6 [==============================] - 0s 664us/sample - loss: 0.0742
    Epoch 152/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0726
    Epoch 153/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0711
    Epoch 154/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0697
    Epoch 155/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0682
    Epoch 156/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0668
    Epoch 157/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0655
    Epoch 158/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0641
    Epoch 159/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0628
    Epoch 160/500
    6/6 [==============================] - 0s 2ms/sample - loss: 0.0615
    Epoch 161/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0603
    Epoch 162/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0590
    Epoch 163/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0578
    Epoch 164/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0566
    Epoch 165/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0555
    Epoch 166/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0543
    Epoch 167/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0532
    Epoch 168/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0521
    Epoch 169/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0510
    Epoch 170/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.0500
    Epoch 171/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0490
    Epoch 172/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0480
    Epoch 173/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0470
    Epoch 174/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0460
    Epoch 175/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0451
    Epoch 176/500
    6/6 [==============================] - 0s 335us/sample - loss: 0.0441
    Epoch 177/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0432
    Epoch 178/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0423
    Epoch 179/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.0415
    Epoch 180/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0406
    Epoch 181/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0398
    Epoch 182/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0390
    Epoch 183/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0382
    Epoch 184/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0374
    Epoch 185/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0366
    Epoch 186/500
    6/6 [==============================] - 0s 497us/sample - loss: 0.0359
    Epoch 187/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0351
    Epoch 188/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0344
    Epoch 189/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0337
    Epoch 190/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0330
    Epoch 191/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0323
    Epoch 192/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0317
    Epoch 193/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0310
    Epoch 194/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0304
    Epoch 195/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0298
    Epoch 196/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0291
    Epoch 197/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0285
    Epoch 198/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0280
    Epoch 199/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0274
    Epoch 200/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0268
    Epoch 201/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0263
    Epoch 202/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0257
    Epoch 203/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0252
    Epoch 204/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0247
    Epoch 205/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0242
    Epoch 206/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0237
    Epoch 207/500
    6/6 [==============================] - 0s 667us/sample - loss: 0.0232
    Epoch 208/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0227
    Epoch 209/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0223
    Epoch 210/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0218
    Epoch 211/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0213
    Epoch 212/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0209
    Epoch 213/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0205
    Epoch 214/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0201
    Epoch 215/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0196
    Epoch 216/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0192
    Epoch 217/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0188
    Epoch 218/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0185
    Epoch 219/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0181
    Epoch 220/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0177
    Epoch 221/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0173
    Epoch 222/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0170
    Epoch 223/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0166
    Epoch 224/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0163
    Epoch 225/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0160
    Epoch 226/500
    6/6 [==============================] - 0s 996us/sample - loss: 0.0156
    Epoch 227/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0153
    Epoch 228/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0150
    Epoch 229/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0147
    Epoch 230/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0144
    Epoch 231/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0141
    Epoch 232/500
    6/6 [==============================] - 0s 2ms/sample - loss: 0.0138
    Epoch 233/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0135
    Epoch 234/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0132
    Epoch 235/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0130
    Epoch 236/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0127
    Epoch 237/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0124
    Epoch 238/500
    6/6 [==============================] - 0s 998us/sample - loss: 0.0122
    Epoch 239/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0119
    Epoch 240/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0117
    Epoch 241/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0115
    Epoch 242/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0112
    Epoch 243/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0110
    Epoch 244/500
    6/6 [==============================] - 0s 830us/sample - loss: 0.0108
    Epoch 245/500
    6/6 [==============================] - 0s 496us/sample - loss: 0.0105
    Epoch 246/500
    6/6 [==============================] - 0s 497us/sample - loss: 0.0103
    Epoch 247/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0101
    Epoch 248/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0099
    Epoch 249/500
    6/6 [==============================] - 0s 664us/sample - loss: 0.0097
    Epoch 250/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0095
    Epoch 251/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0093
    Epoch 252/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.0091
    Epoch 253/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0089
    Epoch 254/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0087
    Epoch 255/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0086
    Epoch 256/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0084
    Epoch 257/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0082
    Epoch 258/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0080
    Epoch 259/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0079
    Epoch 260/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0077
    Epoch 261/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0076
    Epoch 262/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0074
    Epoch 263/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0073
    Epoch 264/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0071
    Epoch 265/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0070
    Epoch 266/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0068
    Epoch 267/500
    6/6 [==============================] - 0s 832us/sample - loss: 0.0067
    Epoch 268/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0065
    Epoch 269/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0064
    Epoch 270/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0063
    Epoch 271/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0061
    Epoch 272/500
    6/6 [==============================] - 0s 2ms/sample - loss: 0.0060
    Epoch 273/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0059
    Epoch 274/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0058
    Epoch 275/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0057
    Epoch 276/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0055
    Epoch 277/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0054
    Epoch 278/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0053
    Epoch 279/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0052
    Epoch 280/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0051
    Epoch 281/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0050
    Epoch 282/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0049
    Epoch 283/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0048
    Epoch 284/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0047
    Epoch 285/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0046
    Epoch 286/500
    6/6 [==============================] - 0s 998us/sample - loss: 0.0045
    Epoch 287/500
    6/6 [==============================] - 0s 500us/sample - loss: 0.0044
    Epoch 288/500
    6/6 [==============================] - 0s 997us/sample - loss: 0.0043
    Epoch 289/500
    6/6 [==============================] - 0s 331us/sample - loss: 0.0042
    Epoch 290/500
    6/6 [==============================] - 0s 335us/sample - loss: 0.0041
    Epoch 291/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0041
    Epoch 292/500
    6/6 [==============================] - 0s 832us/sample - loss: 0.0040
    Epoch 293/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0039
    Epoch 294/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0038
    Epoch 295/500
    6/6 [==============================] - 0s 497us/sample - loss: 0.0037
    Epoch 296/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0037
    Epoch 297/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0036
    Epoch 298/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0035
    Epoch 299/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0034
    Epoch 300/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0034
    Epoch 301/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0033
    Epoch 302/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0032
    Epoch 303/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0032
    Epoch 304/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0031
    Epoch 305/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0030
    Epoch 306/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0030
    Epoch 307/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0029
    Epoch 308/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0029
    Epoch 309/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0028
    Epoch 310/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0027
    Epoch 311/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0027
    Epoch 312/500
    6/6 [==============================] - 0s 829us/sample - loss: 0.0026
    Epoch 313/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0026
    Epoch 314/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0025
    Epoch 315/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0025
    Epoch 316/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0024
    Epoch 317/500
    6/6 [==============================] - 0s 1ms/sample - loss: 0.0024
    Epoch 318/500
    6/6 [==============================] - 0s 664us/sample - loss: 0.0023
    Epoch 319/500
    6/6 [==============================] - 0s 167us/sample - loss: 0.0023
    Epoch 320/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0022
    Epoch 321/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0022
    Epoch 322/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0021
    Epoch 323/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0021
    Epoch 324/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0020
    Epoch 325/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0020
    Epoch 326/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0020
    Epoch 327/500
    6/6 [==============================] - 0s 334us/sample - loss: 0.0019
    Epoch 328/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0019
    Epoch 329/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0018
    Epoch 330/500
    6/6 [==============================] - 0s 997us/sample - loss: 0.0018
    Epoch 331/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0018
    Epoch 332/500
    6/6 [==============================] - 0s 666us/sample - loss: 0.0017
    Epoch 333/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0017
    Epoch 334/500
    6/6 [==============================] - 0s 997us/sample - loss: 0.0017
    Epoch 335/500
    6/6 [==============================] - 0s 997us/sample - loss: 0.0016
    Epoch 336/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0016
    Epoch 337/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0016
    Epoch 338/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0015
    Epoch 339/500
    6/6 [==============================] - 0s 997us/sample - loss: 0.0015
    Epoch 340/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0015
    Epoch 341/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0014
    Epoch 342/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0014
    Epoch 343/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0014
    Epoch 344/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0014
    Epoch 345/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0013
    Epoch 346/500
    6/6 [==============================] - 0s 499us/sample - loss: 0.0013
    Epoch 347/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0013
    Epoch 348/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0012
    Epoch 349/500
    6/6 [==============================] - 0s 166us/sample - loss: 0.0012
    Epoch 350/500
    6/6 [==============================] - 0s 498us/sample - loss: 0.0012
    Epoch 351/500
    6/6 [==============================] - 0s 831us/sample - loss: 0.0012
    Epoch 352/500
    6/6 [==============================] - 0s 333us/sample - loss: 0.0011
    Epoch 353/500
    6/6 [==============================] - 0s 664us/sample - loss: 0.0011
    Epoch 354/500
    6/6 [==============================] - 0s 996us/sample - loss: 0.0011
    Epoch 355/500
    6/6 [==============================] - 0s 665us/sample - loss: 0.0011
    Epoch 356/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0011
    Epoch 357/500
    6/6 [==============================] - 0s 998us/sample - loss: 0.0010
    Epoch 358/500
    6/6 [==============================] - 0s 332us/sample - loss: 0.0010
    Epoch 359/500
    6/6 [==============================] - 0s 663us/sample - loss: 9.8929e-04
    Epoch 360/500
    6/6 [==============================] - 0s 664us/sample - loss: 9.6897e-04
    Epoch 361/500
    6/6 [==============================] - 0s 499us/sample - loss: 9.4906e-04
    Epoch 362/500
    6/6 [==============================] - 0s 665us/sample - loss: 9.2956e-04
    Epoch 363/500
    6/6 [==============================] - 0s 499us/sample - loss: 9.1047e-04
    Epoch 364/500
    6/6 [==============================] - 0s 668us/sample - loss: 8.9177e-04
    Epoch 365/500
    6/6 [==============================] - 0s 498us/sample - loss: 8.7346e-04
    Epoch 366/500
    6/6 [==============================] - 0s 997us/sample - loss: 8.5552e-04
    Epoch 367/500
    6/6 [==============================] - 0s 499us/sample - loss: 8.3794e-04
    Epoch 368/500
    6/6 [==============================] - 0s 665us/sample - loss: 8.2073e-04
    Epoch 369/500
    6/6 [==============================] - 0s 666us/sample - loss: 8.0387e-04
    Epoch 370/500
    6/6 [==============================] - 0s 333us/sample - loss: 7.8736e-04
    Epoch 371/500
    6/6 [==============================] - 0s 503us/sample - loss: 7.7119e-04
    Epoch 372/500
    6/6 [==============================] - 0s 665us/sample - loss: 7.5535e-04
    Epoch 373/500
    6/6 [==============================] - 0s 498us/sample - loss: 7.3983e-04
    Epoch 374/500
    6/6 [==============================] - 0s 998us/sample - loss: 7.2464e-04
    Epoch 375/500
    6/6 [==============================] - 0s 332us/sample - loss: 7.0975e-04
    Epoch 376/500
    6/6 [==============================] - 0s 665us/sample - loss: 6.9517e-04
    Epoch 377/500
    6/6 [==============================] - 0s 665us/sample - loss: 6.8089e-04
    Epoch 378/500
    6/6 [==============================] - 0s 665us/sample - loss: 6.6691e-04
    Epoch 379/500
    6/6 [==============================] - 0s 664us/sample - loss: 6.5321e-04
    Epoch 380/500
    6/6 [==============================] - 0s 3ms/sample - loss: 6.3979e-04
    Epoch 381/500
    6/6 [==============================] - 0s 331us/sample - loss: 6.2665e-04
    Epoch 382/500
    6/6 [==============================] - 0s 1ms/sample - loss: 6.1378e-04
    Epoch 383/500
    6/6 [==============================] - 0s 332us/sample - loss: 6.0117e-04
    Epoch 384/500
    6/6 [==============================] - 0s 333us/sample - loss: 5.8882e-04
    Epoch 385/500
    6/6 [==============================] - 0s 665us/sample - loss: 5.7673e-04
    Epoch 386/500
    6/6 [==============================] - 0s 336us/sample - loss: 5.6488e-04
    Epoch 387/500
    6/6 [==============================] - 0s 498us/sample - loss: 5.5328e-04
    Epoch 388/500
    6/6 [==============================] - 0s 331us/sample - loss: 5.4191e-04
    Epoch 389/500
    6/6 [==============================] - 0s 664us/sample - loss: 5.3078e-04
    Epoch 390/500
    6/6 [==============================] - 0s 495us/sample - loss: 5.1988e-04
    Epoch 391/500
    6/6 [==============================] - 0s 332us/sample - loss: 5.0920e-04
    Epoch 392/500
    6/6 [==============================] - 0s 1ms/sample - loss: 4.9874e-04
    Epoch 393/500
    6/6 [==============================] - 0s 332us/sample - loss: 4.8850e-04
    Epoch 394/500
    6/6 [==============================] - 0s 499us/sample - loss: 4.7846e-04
    Epoch 395/500
    6/6 [==============================] - 0s 828us/sample - loss: 4.6864e-04
    Epoch 396/500
    6/6 [==============================] - 0s 333us/sample - loss: 4.5901e-04
    Epoch 397/500
    6/6 [==============================] - 0s 665us/sample - loss: 4.4958e-04
    Epoch 398/500
    6/6 [==============================] - 0s 665us/sample - loss: 4.4035e-04
    Epoch 399/500
    6/6 [==============================] - 0s 166us/sample - loss: 4.3130e-04
    Epoch 400/500
    6/6 [==============================] - 0s 332us/sample - loss: 4.2244e-04
    Epoch 401/500
    6/6 [==============================] - 0s 332us/sample - loss: 4.1376e-04
    Epoch 402/500
    6/6 [==============================] - 0s 831us/sample - loss: 4.0526e-04
    Epoch 403/500
    6/6 [==============================] - 0s 495us/sample - loss: 3.9694e-04
    Epoch 404/500
    6/6 [==============================] - 0s 499us/sample - loss: 3.8879e-04
    Epoch 405/500
    6/6 [==============================] - 0s 499us/sample - loss: 3.8080e-04
    Epoch 406/500
    6/6 [==============================] - 0s 499us/sample - loss: 3.7298e-04
    Epoch 407/500
    6/6 [==============================] - 0s 333us/sample - loss: 3.6532e-04
    Epoch 408/500
    6/6 [==============================] - 0s 331us/sample - loss: 3.5782e-04
    Epoch 409/500
    6/6 [==============================] - 0s 1ms/sample - loss: 3.5046e-04
    Epoch 410/500
    6/6 [==============================] - 0s 333us/sample - loss: 3.4326e-04
    Epoch 411/500
    6/6 [==============================] - 0s 498us/sample - loss: 3.3622e-04
    Epoch 412/500
    6/6 [==============================] - 0s 666us/sample - loss: 3.2931e-04
    Epoch 413/500
    6/6 [==============================] - 0s 332us/sample - loss: 3.2255e-04
    Epoch 414/500
    6/6 [==============================] - 0s 332us/sample - loss: 3.1592e-04
    Epoch 415/500
    6/6 [==============================] - 0s 498us/sample - loss: 3.0943e-04
    Epoch 416/500
    6/6 [==============================] - 0s 499us/sample - loss: 3.0307e-04
    Epoch 417/500
    6/6 [==============================] - 0s 499us/sample - loss: 2.9685e-04
    Epoch 418/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.9075e-04
    Epoch 419/500
    6/6 [==============================] - 0s 665us/sample - loss: 2.8478e-04
    Epoch 420/500
    6/6 [==============================] - 0s 499us/sample - loss: 2.7893e-04
    Epoch 421/500
    6/6 [==============================] - 0s 498us/sample - loss: 2.7320e-04
    Epoch 422/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.6759e-04
    Epoch 423/500
    6/6 [==============================] - 0s 832us/sample - loss: 2.6209e-04
    Epoch 424/500
    6/6 [==============================] - 0s 665us/sample - loss: 2.5671e-04
    Epoch 425/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.5144e-04
    Epoch 426/500
    6/6 [==============================] - 0s 666us/sample - loss: 2.4627e-04
    Epoch 427/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.4121e-04
    Epoch 428/500
    6/6 [==============================] - 0s 500us/sample - loss: 2.3626e-04
    Epoch 429/500
    6/6 [==============================] - 0s 498us/sample - loss: 2.3140e-04
    Epoch 430/500
    6/6 [==============================] - 0s 666us/sample - loss: 2.2665e-04
    Epoch 431/500
    6/6 [==============================] - 0s 333us/sample - loss: 2.2200e-04
    Epoch 432/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.1744e-04
    Epoch 433/500
    6/6 [==============================] - 0s 334us/sample - loss: 2.1297e-04
    Epoch 434/500
    6/6 [==============================] - 0s 331us/sample - loss: 2.0860e-04
    Epoch 435/500
    6/6 [==============================] - 0s 499us/sample - loss: 2.0431e-04
    Epoch 436/500
    6/6 [==============================] - 0s 332us/sample - loss: 2.0011e-04
    Epoch 437/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.9600e-04
    Epoch 438/500
    6/6 [==============================] - 0s 166us/sample - loss: 1.9198e-04
    Epoch 439/500
    6/6 [==============================] - 0s 167us/sample - loss: 1.8803e-04
    Epoch 440/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.8417e-04
    Epoch 441/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.8039e-04
    Epoch 442/500
    6/6 [==============================] - 0s 331us/sample - loss: 1.7668e-04
    Epoch 443/500
    6/6 [==============================] - 0s 167us/sample - loss: 1.7305e-04
    Epoch 444/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.6950e-04
    Epoch 445/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.6602e-04
    Epoch 446/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.6261e-04
    Epoch 447/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.5927e-04
    Epoch 448/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.5600e-04
    Epoch 449/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.5279e-04
    Epoch 450/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.4965e-04
    Epoch 451/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.4658e-04
    Epoch 452/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.4357e-04
    Epoch 453/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.4062e-04
    Epoch 454/500
    6/6 [==============================] - 0s 999us/sample - loss: 1.3773e-04
    Epoch 455/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.3490e-04
    Epoch 456/500
    6/6 [==============================] - 0s 332us/sample - loss: 1.3213e-04
    Epoch 457/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.2941e-04
    Epoch 458/500
    6/6 [==============================] - 0s 166us/sample - loss: 1.2676e-04
    Epoch 459/500
    6/6 [==============================] - 0s 498us/sample - loss: 1.2415e-04
    Epoch 460/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.2160e-04
    Epoch 461/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.1910e-04
    Epoch 462/500
    6/6 [==============================] - 0s 497us/sample - loss: 1.1666e-04
    Epoch 463/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.1426e-04
    Epoch 464/500
    6/6 [==============================] - 0s 3ms/sample - loss: 1.1191e-04
    Epoch 465/500
    6/6 [==============================] - 0s 499us/sample - loss: 1.0962e-04
    Epoch 466/500
    6/6 [==============================] - 0s 498us/sample - loss: 1.0737e-04
    Epoch 467/500
    6/6 [==============================] - 0s 333us/sample - loss: 1.0516e-04
    Epoch 468/500
    6/6 [==============================] - 0s 166us/sample - loss: 1.0300e-04
    Epoch 469/500
    6/6 [==============================] - 0s 665us/sample - loss: 1.0088e-04
    Epoch 470/500
    6/6 [==============================] - 0s 996us/sample - loss: 9.8813e-05
    Epoch 471/500
    6/6 [==============================] - 0s 664us/sample - loss: 9.6783e-05
    Epoch 472/500
    6/6 [==============================] - 0s 166us/sample - loss: 9.4796e-05
    Epoch 473/500
    6/6 [==============================] - 0s 332us/sample - loss: 9.2848e-05
    Epoch 474/500
    6/6 [==============================] - 0s 332us/sample - loss: 9.0941e-05
    Epoch 475/500
    6/6 [==============================] - 0s 332us/sample - loss: 8.9071e-05
    Epoch 476/500
    6/6 [==============================] - 0s 665us/sample - loss: 8.7242e-05
    Epoch 477/500
    6/6 [==============================] - 0s 333us/sample - loss: 8.5451e-05
    Epoch 478/500
    6/6 [==============================] - 0s 499us/sample - loss: 8.3695e-05
    Epoch 479/500
    6/6 [==============================] - 0s 498us/sample - loss: 8.1977e-05
    Epoch 480/500
    6/6 [==============================] - 0s 332us/sample - loss: 8.0293e-05
    Epoch 481/500
    6/6 [==============================] - 0s 333us/sample - loss: 7.8643e-05
    Epoch 482/500
    6/6 [==============================] - 0s 332us/sample - loss: 7.7028e-05
    Epoch 483/500
    6/6 [==============================] - 0s 332us/sample - loss: 7.5446e-05
    Epoch 484/500
    6/6 [==============================] - 0s 498us/sample - loss: 7.3896e-05
    Epoch 485/500
    6/6 [==============================] - 0s 329us/sample - loss: 7.2379e-05
    Epoch 486/500
    6/6 [==============================] - 0s 498us/sample - loss: 7.0892e-05
    Epoch 487/500
    6/6 [==============================] - 0s 334us/sample - loss: 6.9435e-05
    Epoch 488/500
    6/6 [==============================] - 0s 333us/sample - loss: 6.8008e-05
    Epoch 489/500
    6/6 [==============================] - 0s 499us/sample - loss: 6.6612e-05
    Epoch 490/500
    6/6 [==============================] - 0s 331us/sample - loss: 6.5244e-05
    Epoch 491/500
    6/6 [==============================] - 0s 831us/sample - loss: 6.3903e-05
    Epoch 492/500
    6/6 [==============================] - 0s 498us/sample - loss: 6.2590e-05
    Epoch 493/500
    6/6 [==============================] - 0s 499us/sample - loss: 6.1305e-05
    Epoch 494/500
    6/6 [==============================] - 0s 664us/sample - loss: 6.0046e-05
    Epoch 495/500
    6/6 [==============================] - 0s 332us/sample - loss: 5.8812e-05
    Epoch 496/500
    6/6 [==============================] - 0s 332us/sample - loss: 5.7603e-05
    Epoch 497/500
    6/6 [==============================] - 0s 332us/sample - loss: 5.6420e-05
    Epoch 498/500
    6/6 [==============================] - 0s 333us/sample - loss: 5.5262e-05
    Epoch 499/500
    6/6 [==============================] - 0s 333us/sample - loss: 5.4128e-05
    Epoch 500/500
    6/6 [==============================] - 0s 499us/sample - loss: 5.3016e-05
    




    <tensorflow.python.keras.callbacks.History at 0x238b0d86b38>



Ok, now you have a model that has been trained to learn the relationshop between X and Y. You can use the **model.predict** method to have it figure out the Y for a previously unknown X. So, for example, if X = 10, what do you think Y will be? Take a guess before you run this code:


```python
print(model.predict([10.0]))
```

    [[18.978758]]
    

You might have thought 19, right? But it ended up being a little under. Why do you think that is? 

Remember that neural networks deal with probabilities, so given the data that we fed the NN with, it calculated that there is a very high probability that the relationship between X and Y is Y=2X-1, but with only 6 data points we can't know for sure. As a result, the result for 10 is very close to 19, but not necessarily 19. 

As you work with neural networks, you'll see this pattern recurring. You will almost always deal with probabilities, not certainties, and will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.

