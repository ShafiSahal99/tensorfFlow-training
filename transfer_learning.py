import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5'
base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable = False
new_model = keras.Sequential([
    base_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5)
]
)

new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

new_model.fit(x, y, epochs=15)