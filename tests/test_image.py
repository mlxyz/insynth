import unittest
from unittest import skip

import numpy as np
from PIL import Image
from keras import layers
from tensorflow import keras

from insynth.data import utils
from insynth.metrics.coverage.neuron import NeuronCoverageCalculator
from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageSharpnessPerturbator, \
    ImageContrastPerturbator, ImageFlipPerturbator, ImageOcclusionPerturbator, ImageCompressionPerturbator, \
    ImagePixelizePerturbator


class TestImage(unittest.TestCase):

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

    def _generate_random_image(self):
        imarray = np.random.rand(100, 100, 3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        return im

    def test_ImageNoisePerturbator_without_noise(self):
        input_image = self._generate_random_image()

        perturbator = ImageNoisePerturbator(p=1.0,
                                            noise_prob=type('', (object,), {'rvs': lambda _: 0.0})(),
                                            noise_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_array_equal(input_image, output_image)

    def test_ImageOcclusionPerturbator_without_strength(self):
        input_image = self._generate_random_image()

        perturbator = ImageOcclusionPerturbator(p=1.0, strength_prob=type('', (object,), {'rvs': lambda _: 0})(),
                                                strength_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_array_equal(input_image, output_image)

    def test_ImageOcclusionPerturbator_with_max_strength_and_size(self):
        input_image = self._generate_random_image()
        perturbator = ImageOcclusionPerturbator(p=1.0, strength_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                strength_prob_args={},
                                                width_prob=type('', (object,),
                                                                {'rvs': lambda _: 100, 'mean': lambda _: 100})(),
                                                width_prob_args={},
                                                height_prob=type('', (object,),
                                                                 {'rvs': lambda _: 100, 'mean': lambda _: 100})(),
                                                height_prob_args={})

        output_image = perturbator.apply(input_image)
        # check that at 100% of the image is black
        assert (np.asarray(output_image)[:, :, 0] == 0).sum() == 100 * 100

    def test_ImageOcclusionPerturbator_with_max_strength(self):
        input_image = self._generate_random_image()

        perturbator = ImageOcclusionPerturbator(p=1.0, strength_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                strength_prob_args={},
                                                width_prob=type('', (object,),
                                                                {'rvs': lambda _: 10, 'mean': lambda _: 10})(),
                                                width_prob_args={},
                                                height_prob=type('', (object,),
                                                                 {'rvs': lambda _: 10, 'mean': lambda _: 10})(),
                                                height_prob_args={})

        output_image = perturbator.apply(input_image)
        # check that at least 50% of the image is black
        assert (np.asarray(output_image)[:, :, 0] == 0).sum() > (100 * 100 * 0.5)

    def test_ImageNoisePerturbator_with_noise(self):
        input_image = self._generate_random_image()

        perturbator = ImageNoisePerturbator(p=1.0,
                                            noise_prob=type('', (object,), {'rvs': lambda _: 0.1})(),
                                            noise_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImageBrightnessPerturbator_without_brightness_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageBrightnessPerturbator(p=1.0,
                                                 brightness_change_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                 brightness_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_array_equal(input_image, output_image)

    def test_ImageBrightnessPerturbator_with_brightness_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageBrightnessPerturbator(p=1.0,
                                                 brightness_change_prob=type('', (object,), {'rvs': lambda _: 2.0})(),
                                                 brightness_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImageSharpnessPerturbator_without_sharpness_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageSharpnessPerturbator(p=1.0,
                                                sharpness_change_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                                sharpness_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_array_equal(input_image, output_image)

    def test_ImageSharpnessPerturbator_with_sharpness_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageSharpnessPerturbator(p=1.0,
                                                sharpness_change_prob=type('', (object,), {'rvs': lambda _: 2.0})(),
                                                sharpness_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImageContrastPerturbator_without_contrast_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageContrastPerturbator(p=1.0,
                                               contrast_change_prob=type('', (object,), {'rvs': lambda _: 1.0})(),
                                               contrast_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_array_equal(input_image, output_image)

    def test_ImageContrastPerturbator_with_contrast_change(self):
        input_image = self._generate_random_image()

        perturbator = ImageContrastPerturbator(p=1.0,
                                               contrast_change_prob=type('', (object,), {'rvs': lambda _: 2.0})(),
                                               contrast_change_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImageCompressionPerturbator_with_artifact(self):
        input_image = self._generate_random_image()
        input_image = input_image.convert('RGB')

        perturbator = ImageCompressionPerturbator(p=1.0,
                                                  artifact_prob=type('', (object,), {'rvs': lambda _: 100})(),
                                                  artifact_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImagePixelizePerturbator_with_pixelize(self):
        input_image = self._generate_random_image()
        input_image = input_image.convert('RGB')

        perturbator = ImagePixelizePerturbator(p=1.0,
                                               pixelize_prob=type('', (object,), {'rvs': lambda _: 0.5})(),
                                               pixelize_prob_args={})

        output_image = perturbator.apply(input_image)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, input_image, output_image)

    def test_ImageFlipPerturbator_both(self):
        input_image = self._generate_random_image()

        perturbator = ImageFlipPerturbator(p=1.0, transformation_type='both')

        output_image = perturbator.apply(input_image)
        # check if array was flipped horizontally and vertically
        np.testing.assert_array_equal(output_image, np.flip(input_image, axis=(0, 1)))

    def test_ImageFlipPerturbator_vertically(self):
        input_image = self._generate_random_image()

        perturbator = ImageFlipPerturbator(p=1.0, transformation_type='flip')

        output_image = perturbator.apply(input_image)
        # check if array was flipped vertically
        np.testing.assert_array_equal(output_image, np.flip(input_image, axis=0))

    def test_ImageFlipPerturbator_horizontally(self):
        input_image = self._generate_random_image()

        perturbator = ImageFlipPerturbator(p=1.0, transformation_type='mirror')

        output_image = perturbator.apply(input_image)
        # check if array was flipped horizontally
        np.testing.assert_array_equal(output_image, np.flip(input_image, axis=1))


if __name__ == '__main__':
    unittest.main()
