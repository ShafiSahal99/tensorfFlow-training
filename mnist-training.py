import os

from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


def define_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv2D(32, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Dense(84, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = define_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=30)
model.evaluate(x_test, y_test, batch_size=32)
