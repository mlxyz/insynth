import os
import unittest
from pathlib import Path
from unittest import skip

import librosa
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from insynth.data import utils
from insynth.metrics.coverage.neuron import NeuronCoverageCalculator
from insynth.perturbators.audio import AudioCompressionPerturbator
from insynth.perturbators.image import ImageBrightnessPerturbator, ImageContrastPerturbator, ImageOcclusionPerturbator, \
    ImageCompressionPerturbator
from insynth.runners.runner import ComprehensiveImageRunner, BasicImageRunner, ComprehensiveAudioRunner, \
    ComprehensiveTextRunner


class TestRunner(unittest.TestCase):

    def residual_block(self, x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    def _build_audio_model(self):
        SAMPLING_RATE = 16000
        input_shape = (SAMPLING_RATE // 2, 1)
        num_classes = 5
        inputs = keras.layers.Input(shape=input_shape, name="input")

        x = self.residual_block(inputs, 16, 2)
        x = self.residual_block(x, 32, 2)
        x = self.residual_block(x, 64, 3)
        x = self.residual_block(x, 128, 3)
        x = self.residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

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

    def _build_sentiment_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(10000 + 1, 16),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(1)])

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
        return model

    def test_ComprehensiveImageRunner(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        model = self._generate_mnist_model()
        data_generator = lambda: (Image.fromarray(x) for x in x_test[-10:])
        runner = ComprehensiveImageRunner(
            data_generator, y_test[-10:],
            model, data_generator)
        # filter out compression perturbator as it generates colored images which cannot be processed by this model
        runner.perturbators = [perturbator for perturbator in runner.perturbators if
                               not isinstance(perturbator, ImageCompressionPerturbator)]
        report, robustness = runner.run(save_mutated_samples=False)
        print(report.to_string())
        assert len(report.columns) == 10
        assert report.isna().sum().sum() == 0
        assert isinstance(robustness, float)

    def test_ComprehensiveAudioRunner(self):
        utils.download_and_unzip('https://insynth-data.s3.eu-central-1.amazonaws.com/speaker_recognition.zip',
                                 'data/speaker_recognition/')
        x_test = []
        y_test = []
        for root_dir, directories, files in os.walk('data/speaker_recognition/'):
            if not files:
                continue
            for file in files:
                x_test.append(os.path.join(root_dir, file))
                y_test.append(root_dir)
        data_generator = lambda: (librosa.load(file, sr=None) for file in x_test)
        y_test = LabelEncoder().fit_transform(y_test)
        model = self._build_audio_model()
        runner = ComprehensiveAudioRunner(
            data_generator, y_test,
            model, data_generator)
        runner.perturbators = [AudioCompressionPerturbator(p=1.0)]
        runner.coverage_calculators=[]
        report, robustness = runner.run(save_mutated_samples=False)
        print(report)
        assert len(report.columns) == 7
        assert report.isna().sum().sum() == 0
        assert isinstance(robustness, float)

    def test_ComprehensiveTextRunner(self):
        utils.download_and_unzip('https://insynth-data.s3.eu-central-1.amazonaws.com/sentiment_classification.zip',
                                 'data/sentiment_classification/')
        x_test = []
        y_test = []
        for root_dir, directories, files in os.walk('data/sentiment_classification/'):
            if not files:
                continue
            for file in files:
                x_test.append(os.path.join(root_dir, file))
                y_test.append(root_dir)
        data_generator = lambda: (Path(file).read_text() for file in x_test)
        y_test = LabelEncoder().fit_transform(y_test)
        model = self._build_sentiment_model()
        runner = ComprehensiveTextRunner(data_generator, y_test, model, data_generator)
        report, robustness = runner.run(save_mutated_samples=False)
        print(report.to_string())
        assert len(report.columns) == 10
        assert report.isna().sum().sum() == 0
        assert isinstance(robustness, float)
