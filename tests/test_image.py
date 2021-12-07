import unittest

import numpy as np
from PIL import Image

from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageSharpnessPerturbator, \
    ImageContrastPerturbator, ImageFlipPerturbator, ImageOcclusionPerturbator, ImageCompressionPerturbator, \
    ImagePixelizePerturbator


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
