import unittest

import numpy as np
from PIL import Image
from tensorflow import keras

from insynth.data import utils
from insynth.metrics.coverage.neuron import NeuronCoverageCalculator
from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageSharpnessPerturbator, \
    ImageContrastPerturbator, ImageFlipPerturbator, ImageOcclusionPerturbator, ImageCompressionPerturbator, \
    ImagePixelizePerturbator, DeepXploreImagePerturbator


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

    def test_DeepXploreImagePerturbator(self):
        utils.download_and_unzip('https://insynth-data.s3.eu-central-1.amazonaws.com/imagenette.zip', 'data/imagenet/')
        model1 = keras.applications.MobileNetV2(alpha=0.35, input_shape=(96, 96, 3), include_top=False)
        model2 = keras.applications.MobileNetV2(alpha=0.75, input_shape=(96, 96, 3), include_top=False)
        model3 = keras.applications.MobileNetV2(alpha=1.0, input_shape=(96, 96, 3), include_top=False)
        orig_calc = NeuronCoverageCalculator(model1)
        mut_calc = NeuronCoverageCalculator(model1)
        dataset = keras.utils.image_dataset_from_directory('data/imagenet/', labels=None, batch_size=10000,
                                                           image_size=(96, 96))
        dataset = list(dataset.as_numpy_iterator())[-10:]
        perturbator = DeepXploreImagePerturbator(model1, model2, model3, 'NC')


        for image in dataset:
            orig_calc.update_coverage(image)
            output = perturbator.apply(image, force_mutation=True)
            mut_calc.update_coverage(output)

        orig_coverage = mut_calc.get_coverage()
        mut_coverage = mut_calc.get_coverage()
        pass


if __name__ == '__main__':
    unittest.main()
