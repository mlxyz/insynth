import random
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from insynth.metrics.coverage.neuron import StrongNeuronActivationCoverageCalculator
import numpy as np

from insynth.perturbators.image import DeepXploreImagePerturbator

random.seed(4172306)

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')
x_test /= 255
x_train /= 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model1 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, name='before_softmax'),
        tf.keras.layers.Activation('softmax', name='predictions')
    ]
)

model2 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, name='before_softmax'),
        tf.keras.layers.Activation('softmax', name='predictions')
    ]
)

model3 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, name='before_softmax'),
        tf.keras.layers.Activation('softmax', name='predictions')
    ]
)

batch_size = 128
epochs = 3

model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

deepxplore = DeepXploreImagePerturbator(model1, model2, model3, 'SNAC', x_train[:1000])
coverage_calculator = StrongNeuronActivationCoverageCalculator(model1)
for image in x_train[:1000]:
    coverage_calculator.update_neuron_bounds(np.expand_dims(image, axis=0))
for image in x_test[:10]:
    input_image = np.expand_dims(image, axis=0)
    coverage_calculator.update_coverage(input_image)
    gen_img = deepxplore.apply(input_image)
