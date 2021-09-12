import io
import random

from insynth.input import ImageInput
from insynth.perturbation import BlackboxImagePerturbator, GenericDeepXplorePerturbator, WhiteboxImagePerturbator
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np


class ImageNoisePerturbator(BlackboxImagePerturbator):
    def apply(self, original_input: ImageInput):
        with original_input.image as img:
            image = np.array(img)
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[tuple(coords)] = 0
            return Image.fromarray(out)


class ImageBrightnessPerturbator(BlackboxImagePerturbator):
    def apply(self, original_input: ImageInput):
        with original_input.image as image:
            return ImageEnhance.Brightness(image).enhance(1.5)


class ImageContrastPerturbator(BlackboxImagePerturbator):
    def apply(self, original_input: ImageInput):
        with original_input.image as image:
            return ImageEnhance.Contrast(image).enhance(1.5)


class ImageSharpnessPerturbator(BlackboxImagePerturbator):
    def apply(self, original_input: ImageInput):
        with original_input.image as image:
            return ImageEnhance.Sharpness(image).enhance(1.5)


class ImageOcclusionPerturbator(BlackboxImagePerturbator):
    def __init__(self, probability=0.1, max_width=10, max_height=10, color='#000000'):
        self.probability = probability
        self.max_width = max_width
        self.max_height = max_height
        self.color = color

    def apply(self, original_input: ImageInput):
        average_occlusion_size = (self.max_width / 2) * (self.max_height / 2)
        with original_input.image.copy() as image:
            image_width, image_height = image.size
            number_occlusions = int(image_width * image_height * self.probability / average_occlusion_size)
            draw = ImageDraw.Draw(image)
            for _ in range(number_occlusions):
                occlusion_width = random.randint(1, self.max_width)
                occlusion_height = random.randint(1, self.max_height)
                start_x = random.randint(0, image_width - occlusion_width)
                start_y = random.randint(0, image_height - occlusion_height)
                end_x = start_x + occlusion_width
                end_y = start_y + occlusion_height
                draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=self.color)
            return image


class ImageArtefactPerturbator(BlackboxImagePerturbator):
    def __init__(self, probability=0.2):
        self.probability = probability

    def apply(self, original_input):
        buffer = io.BytesIO()
        with original_input.image as image:
            image.save(buffer, 'JPEG', quality=int(100 - self.probability * 100))
        buffer.flush()
        buffer.seek(0)
        return Image.open(buffer, formats=['JPEG'])


class DeepXploreImagePerturbator(GenericDeepXplorePerturbator, WhiteboxImagePerturbator):

    def apply_gradient_constraint(self, grads):
        return DeepXploreImagePerturbator.constraint_light(grads)

    @staticmethod
    def constraint_occl(gradients, start_point, rect_shape):
        new_grads = np.zeros_like(gradients)
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                         start_point[1]:start_point[1] + rect_shape[1]]
        return new_grads

    @staticmethod
    def constraint_light(gradients):
        new_grads = np.ones_like(gradients)
        grad_mean = np.mean(gradients)
        return grad_mean * new_grads

    @staticmethod
    def constraint_black(gradients, rect_shape=(6, 6)):
        start_point = (
            random.randint(0, gradients.shape[1] - rect_shape[0]),
            random.randint(0, gradients.shape[2] - rect_shape[1]))
        new_grads = np.zeros_like(gradients)
        patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                start_point[1]:start_point[1] + rect_shape[1]]
        if np.mean(patch) < 0:
            new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
            start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
        return new_grads
