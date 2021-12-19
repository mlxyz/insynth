import os

import tensorflow_datasets as tfds
from PIL import Image
from tensorflow import keras

# download data from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and put val into data/imagenette and train into data/imagenette_snac
from insynth.runners.runner import ComprehensiveImageRunner


def ds_generator(ds, size):
    return lambda: (Image.fromarray(sample['image'].numpy()).resize(size).convert('RGB') for sample in
                    ds)


ds = tfds.load('imagenet_v2', split='test', shuffle_files=True)

snac_dataset = ds.take(2000)
test_dataset = ds.skip(2000)

y_test = [sample['label'].numpy() for sample in test_dataset]

print(f'Dataset: Found 8000 images of {len(set(y_test))} classes.')

y_test_snac = [sample['label'].numpy() for sample in snac_dataset]
print(f'SNAC: Found 2000 images of {len(set(y_test_snac))} classes.')

### Xception
xception_model = keras.applications.Xception()

runner = ComprehensiveImageRunner(ds_generator(test_dataset, (299, 299)), y_test, xception_model,
                                  ds_generator(snac_dataset, (299, 299)))

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/xception/')
report.to_csv('output/imagenet/xception/report.csv')

### MobileNetV2
mobilenetv2_model = keras.applications.MobileNetV2()

runner = ComprehensiveImageRunner(ds_generator(test_dataset, (224, 224)), y_test, mobilenetv2_model,
                                  ds_generator(snac_dataset, (224, 224)))

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/mobilenetv2/')
report.to_csv('output/imagenet/mobilenetv2/report.csv')

### InceptionResNetV2

inceptionresnet_model = keras.applications.InceptionResNetV2()

runner = ComprehensiveImageRunner(ds_generator(test_dataset, (299, 299)), y_test, inceptionresnet_model,
                                  ds_generator(snac_dataset, (299, 299)))

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/inceptionresnet/')
report.to_csv('output/imagenet/inceptionresnet/report.csv')
