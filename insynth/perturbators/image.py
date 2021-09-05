from insynth.input import ImageInput
from insynth.perturbation import BlackboxImagePerturbator
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
