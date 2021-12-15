import os

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

### IMAGENET ###
from tqdm import tqdm

# download data from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and put val into data/imagenette and train into data/imagenette_snac
from insynth.runners.runner import ComprehensiveImageRunner

vgg16_model = keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    classifier_activation="softmax"
)

x_test = []
y_test = []

for root_dir, directories, files in os.walk('data/imagenette/'):
    if not files:
        continue
    for file in files:
        image = Image.open(os.path.join(root_dir, file))
        image.load()
        x_test.append(image.convert('RGB'))
        y_test.append(root_dir)

y_test = LabelEncoder().fit_transform(y_test)

print(f'Dataset: Found {len(x_test)} images of {len(set(y_test))} classes.')

print('Resizing images...')
x_test = [x.resize((224, 224)) for x in tqdm(x_test)]

y_test = y_test[-100:]
x_test = x_test[-100:]

x_test_snac = []
y_test_snac = []

for root_dir, directories, files in os.walk('data/imagenette_snac/'):
    if not files:
        continue
    for file in files:
        image = Image.open(os.path.join(root_dir, file))
        image.load()
        x_test_snac.append(image.convert('RGB'))
        y_test_snac.append(root_dir)

y_test_snac = LabelEncoder().fit_transform(y_test_snac)
y_test_snac = y_test_snac[-100:]
x_test_snac = x_test_snac[-100:]

print(f'SNAC: Found {len(x_test_snac)} images of {len(set(y_test_snac))} classes.')

print('Resizing images...')
x_test_snac = [x.resize((224, 224)) for x in tqdm(x_test_snac)]

runner = ComprehensiveImageRunner(x_test, y_test, vgg16_model, x_test_snac)

del x_test_snac
del y_test_snac

report, robustness = runner.run(True, 'output/vgg16_imagenette/')
