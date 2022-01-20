import os

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow import keras

from insynth.perturbators.image import ImageNoisePerturbator, ImageBrightnessPerturbator, ImageContrastPerturbator, \
    ImageSharpnessPerturbator, ImageCompressionPerturbator, ImagePixelizePerturbator, ImageOcclusionPerturbator

np.seterr(divide='ignore', invalid='ignore')
# download data from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and put val into data/imagenette and train into data/imagenette_snac
from insynth.runners.runner import BasicImageRunner


def ds_generator(ds, size):
    return lambda: (Image.fromarray(sample['image'].numpy()).resize(size).convert('RGB') for sample in
                    ds)


ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)

test_dataset = ds.skip(2000).take(1)

y_test = [sample['label'].numpy() for sample in test_dataset]

print(f'Dataset: Found 8000 images of {len(set(y_test))} classes.')

perturbators = [
    *[ImageNoisePerturbator(p=1.0, noise_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                            noise_prob_args={'value': noise}) for noise in
      np.geomspace(0.001, 0.3, 10)],
    *[ImageBrightnessPerturbator(p=1.0,
                                 brightness_change_prob=type('', (object,),
                                                             {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                 brightness_change_prob_args={'value': brightness}) for
      brightness in
      np.linspace(0.05, 3.0, 20)],
    *[ImageContrastPerturbator(p=1.0,
                               contrast_change_prob=type('', (object,),
                                                         {'rvs': lambda _, **kwargs: kwargs['value']})(),
                               contrast_change_prob_args={'value': contrast}) for contrast in
      np.linspace(0.05, 3.0, 20)],
    *[ImageSharpnessPerturbator(p=1.0,
                                sharpness_change_prob=type('', (object,),
                                                           {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                sharpness_change_prob_args={'value': sharpness}) for sharpness in
      np.linspace(0.05, 3.0, 20)],
    *[ImageOcclusionPerturbator(p=1.0,
                                strength_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                strength_prob_args={'value': occlusion}) for occlusion in
      np.geomspace(0.001, 0.5, 10)],
    *[ImageCompressionPerturbator(p=1.0,
                                  artifact_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                                  artifact_prob_args={'value': artifact}) for artifact in
      np.linspace(0.05, 1.0, 10)],
    *[ImagePixelizePerturbator(p=1.0,
                               pixelize_prob=type('', (object,), {'rvs': lambda _, **kwargs: kwargs['value']})(),
                               pixelize_prob_args={'value': pixelize}) for pixelize in
      np.linspace(0.05, 0.95, 10)]
]



### Xception
xception_model = keras.applications.Xception()

runner = BasicImageRunner(perturbators, [], ds_generator(test_dataset, (299, 299)), y_test, xception_model,
                          lambda sample: keras.applications.xception.preprocess_input(
                              np.expand_dims(np.array(sample), axis=0)))
os.makedirs('output/imagenet/xception/')
report, robustness = runner.run(True, 'output/imagenet/xception')

print(report.to_string())
print(robustness)

report.to_csv('output/imagenet/xception/report.csv')

### MobileNetV2
mobilenetv2_model = keras.applications.MobileNetV2()

runner = BasicImageRunner(perturbators, [], ds_generator(test_dataset, (224, 224)), y_test, mobilenetv2_model,
                          lambda sample: keras.applications.mobilenet_v2.preprocess_input(
                              np.expand_dims(np.array(sample), axis=0)))
os.makedirs('output/imagenet/mobilenetv2/')
report, robustness = runner.run(True, 'output/imagenet/mobilenetv2')

print(report.to_string())
print(robustness)

report.to_csv('output/imagenet/mobilenetv2/report.csv')

### InceptionResNetV2

inceptionresnet_model = keras.applications.InceptionResNetV2()

runner = BasicImageRunner(perturbators, [], ds_generator(test_dataset, (299, 299)), y_test, inceptionresnet_model,
                          lambda sample: keras.applications.inception_resnet_v2.preprocess_input(
                              np.expand_dims(np.array(sample), axis=0)))

os.makedirs('output/imagenet/inceptionresnet/')
report, robustness = runner.run(True, 'output/imagenet/inceptionresnet')

print(report.to_string())
print(robustness)

report.to_csv('output/imagenet/inceptionresnet/report.csv')
