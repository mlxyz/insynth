import os
import pickle
import re
import string
from itertools import tee

import tensorflow as tf
# download and extract https://insynth-data.s3.eu-central-1.amazonaws.com/sentiment_analysis_experiment.zip to data/
from keras.layers import TextVectorization

from insynth.runners.runner import ComprehensiveTextRunner

DATA_ROOT_PATH = '../data/sentiment_analysis/'
batch_size = 32
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
    shuffle=True
).unbatch()
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    shuffle=True
).unbatch()
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    DATA_ROOT_PATH + "aclImdb/test", batch_size=batch_size, shuffle=True
).unbatch()

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

for text_sample, label_sample in raw_train_ds.take(5):
    print(text_sample.numpy())
    print(label_sample.numpy())


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


model = tf.keras.models.load_model(DATA_ROOT_PATH + 'model.h5')
# bug see https://stackoverflow.com/questions/70255845/tensorflow-textvectorization-producing-ragged-tensor-with-no-padding-after-loadi
from_disk = pickle.load(open(DATA_ROOT_PATH + "tv_layer.pkl", "rb"))
text_vectroize_layer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                         output_mode='int',
                                         output_sequence_length=from_disk['config']['output_sequence_length'])

text_vectroize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
text_vectroize_layer.set_weights(from_disk['weights'])


def vectorize_text(text):
    return text_vectroize_layer([text])

x_iterator,y_iterator = tee(raw_val_ds.as_numpy_iterator())
x_test_data_generator = lambda: (sample[0].decode('utf-8') for sample in x_iterator)
y_test = [sample[1] for sample in y_iterator]
x_snac_data_generator = lambda: (sample[0].numpy().decode('utf-8') for sample in raw_train_ds.take(100))
model.summary()
runner = ComprehensiveTextRunner(x_test_data_generator, y_test, model,
                                 x_snac_data_generator,
                                 pre_predict_lambda=vectorize_text)
os.makedirs('output/sentiment_analysis/')
report, robustness = runner.run(True, 'output/sentiment_analysis/')

print(report.to_string())
print(robustness)

report.to_csv('output/sentiment_analysis/report.csv')
