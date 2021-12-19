import os

import tensorflow_datasets as tfds
from PIL import Image
from tensorflow import keras

# download data from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and put val into data/imagenette and train into data/imagenette_snac
from insynth.runners.runner import ComprehensiveImageRunner

ds = tfds.load('imagenet_v2', split='test', shuffle_files=True)

snac_dataset = ds.take(2000)
test_dataset = ds.skip(2000)

x_test_generator = lambda: (Image.fromarray(sample['image'].numpy()).resize((224, 224)).convert('RGB') for sample in
                            test_dataset)
y_test = [sample['label'] for sample in test_dataset]

print(f'Dataset: Found {len(test_dataset)} images of {len(set(y_test))} classes.')

x_test_snac_generator = lambda: (Image.fromarray(sample['image'].numpy()).resize((224, 224)).convert('RGB') for sample
                                 in
                                 snac_dataset)
y_test_snac = [sample['label'] for sample in snac_dataset]
print(f'SNAC: Found {len(snac_dataset)} images of {len(set(y_test_snac))} classes.')

### Xception
xception_model = keras.applications.Xception()

runner = ComprehensiveImageRunner(x_test_generator, y_test, xception_model, x_test_snac_generator)

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/xception/')
report.to_csv('output/imagenet/xception/report.csv')

### MobileNetV2
mobilenetv2_model = keras.applications.MobileNetV2()

runner = ComprehensiveImageRunner(x_test_generator, y_test, mobilenetv2_model, x_test_snac_generator)

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/mobilenetv2/')
report.to_csv('output/imagenet/mobilenetv2/report.csv')

### InceptionResNetV2

inceptionresnet_model = keras.applications.InceptionResNetV2()

runner = ComprehensiveImageRunner(x_test_generator, y_test, inceptionresnet_model, x_test_snac_generator)

report, robustness = runner.run()

print(report.to_string())
print(robustness)

os.makedirs('output/imagenet/inceptionresnet/')
report.to_csv('output/imagenet/inceptionresnet/report.csv')
