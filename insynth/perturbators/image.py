import random

from insynth.input import ImageInput
from insynth.perturbation import BlackboxImagePerturbator, GenericDeepXplorePerturbator, WhiteboxImagePerturbator
from PIL import Image, ImageEnhance
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
