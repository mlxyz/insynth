import unittest

import numpy as np
from PIL import Image

from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageSharpnessPerturbator, \
    ImageContrastPerturbator


class TestImage(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
