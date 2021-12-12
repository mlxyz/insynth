import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from insynth.metrics.coverage.neuron import NeuronCoverageCalculator
from insynth.perturbators.image import ImageBrightnessPerturbator, ImageContrastPerturbator, ImageOcclusionPerturbator, \
    ImageCompressionPerturbator
from insynth.runners.runner import ComprehensiveImageRunner, BasicImageRunner


class TestRunner(unittest.TestCase):
    def _generate_mnist_model(self):
        num_classes = 10
        input_shape = (28, 28, 1)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        x_train = np.expand_dims(x_train, -1)

        y_train = keras.utils.to_categorical(y_train, num_classes)

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.summary()
        batch_size = 32
        epochs = 15
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # dont fit the model for CI/CD
        # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return model

    def test_BasicImageRunner(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        model = self._generate_mnist_model()
        runner = BasicImageRunner(
            [ImageBrightnessPerturbator(p=1.0), ImageContrastPerturbator(p=1.0),
             ImageOcclusionPerturbator(p=1.0, width_prob_args={'loc': 2, 'scale': 2},
                                       height_prob_args={'loc': 2, 'scale': 2},
                                       strength_prob_args={'loc': 0.1, 'scale': 0.05})],
            [NeuronCoverageCalculator(model)], x_test[-1000:], y_test[-1000:],
            model)
        report = runner.run(save_images=False)
        assert len(report.columns) == 7
        assert report.isna().sum().sum() == 0

    def test_ComprehensiveImageRunner(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        model = self._generate_mnist_model()
        runner = ComprehensiveImageRunner(
            x_test[-1000:], y_test[-1000:],
            model)
        # filter out compression perturbator as it generates colored images which cannot be processed by this model
        runner.perturbators = [perturbator for perturbator in runner.perturbators if
                               not isinstance(perturbator, ImageCompressionPerturbator)]
        report = runner.run(save_images=False)
        print(report)
        assert len(report.columns) == 7
        assert report.isna().sum().sum() == 0
