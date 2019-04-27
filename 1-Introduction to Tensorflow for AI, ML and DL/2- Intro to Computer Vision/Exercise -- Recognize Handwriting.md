
## Exercise 2
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. 

Here's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:
1. It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

I've started the code for you below -- how would you finish it? 


```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# YOUR CODE SHOULD START HERE
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# YOUR CODE SHOULD START HERE

x_train=x_train/255.0
x_test=x_test/255.0

callbacks = myCallback()

# YOUR CODE SHOULD END HERE
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# YOUR CODE SHOULD START HERE
model.fit(x_train, y_train, callbacks=[callbacks])
# YOUR CODE SHOULD END HERE

model.evaluate(x_test, y_test)
```

    60000/60000 [==============================] - 6s 106us/sample - loss: 0.2588 - acc: 0.9255
    10000/10000 [==============================] - 1s 52us/sample - loss: 0.1368 - acc: 0.9594
    




    [0.13679398454613984, 0.9594]


