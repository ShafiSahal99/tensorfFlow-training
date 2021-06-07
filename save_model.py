import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

model1 = keras.Sequential([layers.Dense(64, activation="relu"), layers.Dense(10)])

inputs = keras.Input(784)
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)


# SavedModel format or HDF5 format
model = keras.models.load_model('complete_saved_model/')
# model = keras.models.load_model('saved_model/')
# model.load_weights('checkpoint_folder/')

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=2)
model.evaluate(x_test, y_test, batch_size=32)
model.save('complete_saved_model/')
