import os
import random

from PIL import Image
from tensorflow import keras

# download data from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and put val into data/imagenette and train into data/imagenette_snac
from insynth.runners.runner import ComprehensiveImageRunner

path_to_class = {
    'data/imagenette/n01440764': 0,
    'data/imagenette/n02102040': 217,
    'data/imagenette/n02979186': 482,
    'data/imagenette/n03000684': 491,
    'data/imagenette/n03028079': 497,
    'data/imagenette/n03394916': 566,
    'data/imagenette/n03417042': 569,
    'data/imagenette/n03425413': 571,
    'data/imagenette/n03445777': 574,
    'data/imagenette/n03888257': 701,

    'data/imagenette_snac/n01440764': 0,
    'data/imagenette_snac/n02102040': 217,
    'data/imagenette_snac/n02979186': 482,
    'data/imagenette_snac/n03000684': 491,
    'data/imagenette_snac/n03028079': 497,
    'data/imagenette_snac/n03394916': 566,
    'data/imagenette_snac/n03417042': 569,
    'data/imagenette_snac/n03425413': 571,
    'data/imagenette_snac/n03445777': 574,
    'data/imagenette_snac/n03888257': 701
}

x_test = []
y_test = []

for root_dir, directories, files in os.walk('data/imagenette/'):
    if not files:
        continue
    for file in files:
        x_test.append(os.path.join(root_dir, file))
        y_test.append(path_to_class[root_dir])

random.Random(12345).shuffle(x_test)
random.Random(12345).shuffle(y_test)
x_test = x_test[:100]
y_test = y_test[:100]

x_test_generator = lambda: (Image.open(file).resize((224, 224)).convert('RGB') for file in x_test)

print(f'Dataset: Found {len(x_test)} images of {len(set(y_test))} classes.')

x_test_snac = []
y_test_snac = []

for root_dir, directories, files in os.walk('data/imagenette_snac/'):
    if not files:
        continue
    for file in files:
        x_test_snac.append(os.path.join(root_dir, file))
        y_test_snac.append(path_to_class[root_dir])
random.Random(12345).shuffle(x_test_snac)
random.Random(12345).shuffle(y_test_snac)
x_test_snac = x_test_snac[:100]
y_test_snac = y_test_snac[:100]

x_test_snac_generator = lambda: (Image.open(file).resize((224, 224)).convert('RGB') for file in x_test_snac)

print(f'SNAC: Found {len(x_test_snac)} images of {len(set(y_test_snac))} classes.')




### Xception
xception_model = keras.applications.Xception()

runner = ComprehensiveImageRunner(x_test_generator, y_test, xception_model, x_test_snac_generator)

report, robustness = runner.run(False, 'output/vgg16_imagenette/')

print(report.to_string())
print(robustness)

os.makedirs('output/imagenette/vgg16/')
report.to_csv('output/imagenette/vgg16/report.csv')


### MobileNetV2
mobilenetv2_model = keras.applications.MobileNetV2()

runner = ComprehensiveImageRunner(x_test_generator, y_test, mobilenetv2_model, x_test_snac_generator)

report, robustness = runner.run(False, 'output/mobilenetv2_model_imagenette/')

print(report.to_string())
print(robustness)

os.makedirs('output/imagenette/mobilenetv2_model/')
report.to_csv('output/imagenette/mobilenetv2_model/report.csv')


### InceptionResNetV2

inceptionresnet_model = keras.applications.InceptionResNetV2()

runner = ComprehensiveImageRunner(x_test_generator, y_test, inceptionresnet_model, x_test_snac_generator)

report, robustness = runner.run(False, 'output/inceptionresnet_model_model_imagenette/')

print(report.to_string())
print(robustness)

os.makedirs('output/imagenette/inceptionresnet_model_model/')
report.to_csv('output/imagenette/inceptionresnet_model_model/report.csv')