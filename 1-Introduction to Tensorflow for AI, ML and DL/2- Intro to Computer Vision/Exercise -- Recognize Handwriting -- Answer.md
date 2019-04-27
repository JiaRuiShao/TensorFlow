

```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist

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
```

    C:\Users\surface\ananew\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    

    WARNING:tensorflow:From C:\Users\surface\ananew\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    Epoch 1/10
    60000/60000 [==============================] - 24s 397us/sample - loss: 0.2011 - acc: 0.9408
    Epoch 2/10
    60000/60000 [==============================] - 22s 373us/sample - loss: 0.0801 - acc: 0.9751
    Epoch 3/10
    60000/60000 [==============================] - 22s 360us/sample - loss: 0.0533 - acc: 0.9826
    Epoch 4/10
    60000/60000 [==============================] - 25s 425us/sample - loss: 0.0363 - acc: 0.9879
    Epoch 5/10
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0269 - acc: 0.9916
    Reached 99% accuracy so cancelling training!
    60000/60000 [==============================] - 26s 430us/sample - loss: 0.0269 - acc: 0.9916
    




    <tensorflow.python.keras.callbacks.History at 0x1dcc568a588>


