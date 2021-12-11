import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageOps
from scipy.stats import norm

from insynth.perturbation import BlackboxImagePerturbator, WhiteboxImagePerturbator


class ImageNoisePerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, noise_prob=norm, noise_prob_args={'loc': 0.01, 'scale': 0.005}):
        super().__init__(p)
        self.noise_prob = noise_prob
        self.noise_prob_args = noise_prob_args

    def _internal_apply(self, original_input: Image):
        if random.random() > self.p:
            return original_input
        with original_input as img:
            image = np.array(img)
            salt_pepper_ratio = 0.5
            amount = max(self.noise_prob.rvs(**self.noise_prob_args), 0.0)
            output_image_arr = np.copy(image)
            # Salt
            num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            output_image_arr[tuple(coords)] = 1

            # Pepper
            num_pepper = np.ceil(amount * image.size * (1. - salt_pepper_ratio))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            output_image_arr[tuple(coords)] = 0
            return Image.fromarray(output_image_arr)


class ImageBrightnessPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, brightness_change_prob=norm, brightness_change_prob_args={'loc': 1, 'scale': 0.5}):
        super().__init__(p)
        self.brightness_change_prob = brightness_change_prob
        self.brightness_change_prob_args = brightness_change_prob_args

    def _internal_apply(self, original_input: Image):
        with original_input as image:
            return ImageEnhance.Brightness(image).enhance(
                self.brightness_change_prob.rvs(**self.brightness_change_prob_args))


class ImageContrastPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, contrast_change_prob=norm, contrast_change_prob_args={'loc': 1, 'scale': 0.5}):
        super().__init__(p)
        self.contrast_change_prob = contrast_change_prob
        self.contrast_change_prob_args = contrast_change_prob_args

    def _internal_apply(self, original_input: Image):
        with original_input as image:
            return ImageEnhance.Contrast(image).enhance(self.contrast_change_prob.rvs(**self.contrast_change_prob_args))


class ImageSharpnessPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, sharpness_change_prob=norm, sharpness_change_prob_args={'loc': 1, 'scale': 0.5}):
        super().__init__(p)
        self.sharpness_change_prob = sharpness_change_prob
        self.sharpness_change_prob_args = sharpness_change_prob_args

    def _internal_apply(self, original_input: Image):
        with original_input as image:
            return ImageEnhance.Sharpness(image).enhance(
                self.sharpness_change_prob.rvs(**self.sharpness_change_prob_args))


class ImageFlipPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, transformation_type='mirror'):
        super().__init__(p)
        self.transformation_type = transformation_type

    def _internal_apply(self, original_input: Image):
        with original_input.copy() as image:
            if (self.transformation_type == 'flip' or self.transformation_type == 'both') and random.random() < self.p:
                image = ImageOps.flip(image)
            if (self.transformation_type == 'mirror' or self.transformation_type == 'both') \
                    and random.random() < self.p:
                image = ImageOps.mirror(image)
            return image


class ImageOcclusionPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, strength_prob=norm, strength_prob_args={'loc': 0.2, 'scale': 0.05}, width_prob=norm,
                 width_prob_args={'loc': 10, 'scale': 5}, height_prob=norm,
                 height_prob_args={'loc': 10, 'scale': 5}, color='#000000'):
        super().__init__(p)
        self.strength_prob = strength_prob
        self.strength_prob_args = strength_prob_args
        self.width_prob = width_prob
        self.width_prob_args = width_prob_args
        self.height_prob = height_prob
        self.height_prob_args = height_prob_args
        self.color = color

    def _internal_apply(self, original_input: Image):
        with original_input.copy() as image:
            image_width, image_height = image.size
            strength = self.strength_prob.rvs(**self.strength_prob_args)
            number_occlusions = int((image_width * image_height * strength) / (
                    self.width_prob.mean(**self.width_prob_args) * self.height_prob.mean(**self.height_prob_args)))
            draw = ImageDraw.Draw(image)
            for _ in range(number_occlusions):
                occlusion_width = int(self.width_prob.rvs(**self.width_prob_args))
                occlusion_height = int(self.height_prob.rvs(**self.height_prob_args))
                start_x = random.randint(0, max(image_width - occlusion_width, 0))
                start_y = random.randint(0, max(image_height - occlusion_height, 0))
                end_x = start_x + occlusion_width
                end_y = start_y + occlusion_height
                draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=self.color)
            return image


class ImageCompressionPerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, artifact_prob=norm, artifact_prob_args={'loc': 0.5, 'scale': 0.2}):
        super().__init__(p)
        self.artifact_prob = artifact_prob
        self.artifact_prob_args = artifact_prob_args

    def _internal_apply(self, original_input: Image):
        buffer = io.BytesIO()
        with original_input as image:
            image.convert('RGB').save(buffer, 'JPEG',
                                      quality=int(100 - self.artifact_prob.rvs(**self.artifact_prob_args) * 100))
        buffer.flush()
        buffer.seek(0)
        return Image.open(buffer, formats=['JPEG']).convert('RGB')


class ImagePixelizePerturbator(BlackboxImagePerturbator):
    def __init__(self, p=0.5, pixelize_prob=norm, pixelize_prob_args={'loc': 0.2, 'scale': 0.1}):
        super().__init__(p)
        self.pixelize_prob = pixelize_prob
        self.pixelize_prob_args = pixelize_prob_args

    def _internal_apply(self, original_input: Image):
        with original_input as image:
            image_width, image_height = image.size
            pixelize_factor = self.pixelize_prob.rvs(**self.pixelize_prob_args)
            image_small = image.resize(
                (int(image_width * (1 - pixelize_factor)), int(image_height * (1 - pixelize_factor))),
                resample=Image.BILINEAR)
            return image_small.resize(image.size, Image.NEAREST)

