
## Exercise 2

In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"


```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

callbacks = myCallback()

# build a DNN model with a single convolutional layer and a single MaxPooling 2D
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])

test_loss, test_acc = model.evaluate(test_images, test_labels)
```

    C:\Users\surface\ananew\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    Epoch 1/20
    60000/60000 [==============================] - 40s 660us/sample - loss: 0.1683 - acc: 0.9510
    Epoch 2/20
    60000/60000 [==============================] - 44s 734us/sample - loss: 0.0566 - acc: 0.9828
    Epoch 3/20
    60000/60000 [==============================] - 38s 636us/sample - loss: 0.0383 - acc: 0.9880
    Epoch 4/20
    60000/60000 [==============================] - 38s 638us/sample - loss: 0.0255 - acc: 0.9919
    Epoch 5/20
    60000/60000 [==============================] - 38s 625us/sample - loss: 0.0168 - acc: 0.9944
    Epoch 6/20
    60000/60000 [==============================] - 38s 636us/sample - loss: 0.0125 - acc: 0.9960
    Epoch 7/20
    60000/60000 [==============================] - 38s 629us/sample - loss: 0.0095 - acc: 0.9969
    Epoch 8/20
    60000/60000 [==============================] - 38s 632us/sample - loss: 0.0077 - acc: 0.9973
    Epoch 9/20
    60000/60000 [==============================] - 40s 664us/sample - loss: 0.0070 - acc: 0.9976
    Epoch 10/20
    59904/60000 [============================>.] - ETA: 0s - loss: 0.0041 - acc: 0.9987
    Reached 99.8% accuracy so cancelling training!
    60000/60000 [==============================] - 39s 649us/sample - loss: 0.0041 - acc: 0.9987
    10000/10000 [==============================] - 2s 221us/sample - loss: 0.0496 - acc: 0.9875
    
